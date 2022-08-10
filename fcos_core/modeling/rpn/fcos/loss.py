"""
This file contains specific functions for computing losses of FCOS
file
"""

import torch
from torch.nn import functional as F
from torch import nn
import os
from ..utils import concat_box_prediction_layers
from fcos_core.layers import IOULoss
from fcos_core.layers import SigmoidFocalLoss
from fcos_core.layers import GaussianFocalLoss
from fcos_core.modeling.matcher import Matcher
from fcos_core.modeling.utils import cat
from fcos_core.structures.boxlist_ops import boxlist_iou
from fcos_core.structures.boxlist_ops import cat_boxlist
from .assign import make_assigner
from fcos_core.layers import smooth_l1_loss
from fcos_core.layers import SEPFocalLoss
import numpy as np


INF = 100000000


def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    return tensor


class FCOSLossComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        self.cls_loss_func = SigmoidFocalLoss(
            cfg.MODEL.FCOS.LOSS_GAMMA,
            cfg.MODEL.FCOS.LOSS_ALPHA
        )
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        # tricks
        self.center_sampling_radius = cfg.MODEL.FCOS.CENTER_SAMPLING_RADIUS
        self.iou_loss_type = cfg.MODEL.FCOS.IOU_LOSS_TYPE
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS

        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.box_reg_loss_func = IOULoss(self.iou_loss_type)
        self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        self.corners_cls_loss_func = GaussianFocalLoss(
            alpha=2.0,
            gamma=4.0,
            loss_weight=0.25
        )
        self.corners_reg_loss_func = smooth_l1_loss
        
        self.corners_assigner = make_assigner(
            gaussian_bump=True,
            gaussian_iou=0.7
        )
        self.fg_loss_func = SEPFocalLoss(
            gamma=2.0, 
            alpha=0.25, 
            loss_weight=0.1
        )
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1

        # we use BCE to select which polar coords to use
        self.cos_loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        self.rou_loss_func = nn.L1Loss(reduction="sum")

    def get_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys, radius=1.0):
        '''
        This code is from
        https://github.com/yqyao/FCOS_PLUS/blob/0d20ba34ccc316650d8c30febb2eb40cb6eaae37/
        maskrcnn_benchmark/modeling/rpn/fcos/loss.py#L42
        '''
        num_gts = gt.shape[0]
        K = len(gt_xs)
        gt = gt[None].expand(K, num_gts, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape) # P G 4
        # no gt
        if center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(
                xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0]
            )
            center_gt[beg:end, :, 1] = torch.where(
                ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1]
            )
            center_gt[beg:end, :, 2] = torch.where(
                xmax > gt[beg:end, :, 2],
                gt[beg:end, :, 2], xmax
            )
            center_gt[beg:end, :, 3] = torch.where(
                ymax > gt[beg:end, :, 3],
                gt[beg:end, :, 3], ymax
            )
            beg = end
        left = gt_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs[:, None]
        top = gt_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def prepare_targets(self, points, targets):
        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ]
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )

        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        self.num_points_per_level = num_points_per_level
        points_all_level = torch.cat(points, dim=0)
        labels, reg_targets = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest
        )

        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )

            reg_targets_per_level = torch.cat([
                reg_targets_per_im[level]
                for reg_targets_per_im in reg_targets
            ], dim=0)

            if self.norm_reg_targets:
                reg_targets_per_level = reg_targets_per_level / self.fpn_strides[level]
            reg_targets_level_first.append(reg_targets_per_level)

        return labels_level_first, reg_targets_level_first

    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            bboxes = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")
            area = targets_per_im.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            rou1 = torch.sqrt(l*l + t*t) # P G
            rou2 = torch.sqrt(r*r + b*b)

            rou3 = torch.sqrt(r*r + t*t)
            rou4 = torch.sqrt(l*l + b*b)
            rou_per_im = torch.stack([rou1, rou2], dim=2) # P G 2
            ratio_1 = torch.min(rou1, rou2)/torch.max(rou1, rou2)
            #ratio_2 = torch.min(rou3, rou4)/torch.max(rou3, rou4)
            #is_balance = (abs(ratio_1-ratio_2) <= 0.5)
            lr_max = torch.max(abs(l), abs(r))
            lr_min = torch.min(abs(l), abs(r))
            tb_max = torch.max(abs(t), abs(b))
            tb_min = torch.min(abs(t), abs(b))
            center = torch.sqrt(lr_min*tb_min/(lr_max*tb_max))

            is_ok = (center>0.4)
            is_in = reg_targets_per_im.min(dim=2)[0] > 0
            is_balance = is_ok & is_in

            if self.center_sampling_radius > 0:
                is_in_boxes = self.get_sample_region(
                    bboxes,
                    self.fpn_strides,
                    self.num_points_per_level,
                    xs, ys,
                    radius=self.center_sampling_radius
                )
                # is_in_boxes = is_in_boxes | is_balance
            else:
                # no center sampling, it will use all the locations within a ground-truth box
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            #max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            max_reg_targets_per_im = rou_per_im.max(dim=2)[0]
            
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF
            ##
            locations_to_gt_area[is_balance == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = 0

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return labels, reg_targets

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def compute_pair_prob_targets(self, reg_targets):
        # reg_targets: P 4
        l, t, r, b = reg_targets.chunk(4, dim=-1)
        l = l.squeeze()
        t = t.squeeze()
        r = r.squeeze()
        b = b.squeeze()
        # tl
        rou1 = torch.sqrt(t*t + l*l)
        sin_theta_1 = t / rou1
        rou2 = torch.sqrt(b*b + r*r)
        sin_theta_2 = r / rou2

        centerness_targets = torch.min(rou1, rou2)/torch.max(rou1, rou2)

        rou_pred_12 = torch.stack([rou1, rou2], dim=-1)
        theta_pred_12 = torch.stack([sin_theta_1, sin_theta_2], dim=-1)
        
        # we design rou & theta
        rou_targets = reg_targets.new_zeros(reg_targets.shape[0], 2)
        theta_targets = reg_targets.new_zeros(reg_targets.shape[0], 2)
        rou_targets = rou_pred_12
        theta_targets = theta_pred_12

        return centerness_targets, rou_targets, theta_targets

    def compute_corner_targets_for_locations(self, points, targets, strides):
        #
        assert points.shape[0] == strides.shape[0]
        assigner = self.corners_assigner
        gt_hm_tl, gt_offset_tl, pos_inds_tl, neg_inds_tl, \
        gt_hm_br, gt_offset_br, pos_inds_br, neg_inds_br = \
            assigner.assign(points, targets, strides)

        num_valid_points = points.shape[0]
        hm_tl_weights = points.new_zeros(num_valid_points, dtype=torch.float)
        hm_br_weights = points.new_zeros(num_valid_points, dtype=torch.float)
        offset_tl_weights = points.new_zeros([num_valid_points, 2], dtype=torch.float)
        offset_br_weights = points.new_zeros([num_valid_points, 2], dtype=torch.float)

        hm_tl_weights[pos_inds_tl] = 1.0
        hm_tl_weights[neg_inds_tl] = 1.0
        offset_tl_weights[pos_inds_tl, :] = 1.0

        hm_br_weights[pos_inds_br] = 1.0
        hm_br_weights[neg_inds_br] = 1.0
        offset_br_weights[pos_inds_br, :] = 1.0

        return (gt_hm_tl, gt_offset_tl, hm_tl_weights, offset_tl_weights, pos_inds_tl, neg_inds_tl,
                gt_hm_br, gt_offset_br, hm_br_weights, offset_br_weights, pos_inds_br, neg_inds_br)
        
    def prepare_corner_targets(self, points, targets):
        expanded_strides_per_level = []
        for l, points_per_level in enumerate(points):
            strides_per_level = \
                points_per_level.new_tensor(self.fpn_strides[l])
            expanded_strides_per_level.append(
                strides_per_level[None].expand(len(points_per_level), -1)
            )

        expanded_strides_per_level = torch.cat(expanded_strides_per_level, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        points_all_level = torch.cat(points, dim=0)
        #[]*N tl
        gt_hm_tl = []
        gt_offset_tl = [] 
        hm_tl_weights = [] 
        offset_tl_weights = [] 
        pos_inds_tl = []
        # br
        gt_hm_br = [] 
        gt_offset_br = [] 
        hm_br_weights = [] 
        offset_br_weights = [] 
        pos_inds_br = [] 
        for target in targets:
            g_tl, o_tl, wg_tl, wo_tl, p_ind_tl, _, \
            g_br, o_br, wg_br, wo_br, p_ind_br, _ = self.compute_corner_targets_for_locations(
                points_all_level, target, expanded_strides_per_level
            )
            gt_hm_tl.append(g_tl)
            gt_offset_tl.append(o_tl)
            hm_tl_weights.append(wg_tl)
            offset_tl_weights.append(wo_tl)
            pos_inds_tl.append(p_ind_tl)

            gt_hm_br.append(g_br)
            gt_offset_br.append(o_br)
            hm_br_weights.append(wg_br)
            offset_br_weights.append(wo_br)
            pos_inds_br.append(p_ind_br)

        # img -> level
        for i in range(len(gt_hm_tl)):
            # tl
            gt_hm_tl[i] = torch.split(gt_hm_tl[i], num_points_per_level, dim=0)
            gt_offset_tl[i] = torch.split(gt_offset_tl[i], num_points_per_level, dim=0)
            hm_tl_weights[i] = torch.split(hm_tl_weights[i], num_points_per_level, dim=0)
            offset_tl_weights[i] = torch.split(offset_tl_weights[i], num_points_per_level, dim=0)
            #br
            gt_hm_br[i] = torch.split(gt_hm_br[i], num_points_per_level, dim=0)
            gt_offset_br[i] = torch.split(gt_offset_br[i], num_points_per_level, dim=0)
            hm_br_weights[i] = torch.split(hm_br_weights[i], num_points_per_level, dim=0)
            offset_br_weights[i] = torch.split(offset_br_weights[i], num_points_per_level, dim=0)

        gt_hm_tl_list = []
        gt_offset_tl_list = []
        hm_tl_weight_list = []
        offset_tl_weight_list = []
        # br
        gt_hm_br_list = []
        gt_offset_br_list = []
        hm_br_weight_list = []
        offset_br_weight_list = []

        for level in range(len(points)):
            gt_hm_tl_list.append(
                torch.cat([gt_hm_tl_per_im[level] for gt_hm_tl_per_im in gt_hm_tl], dim=0)
            )
            gt_offset_tl_list.append(
                torch.cat([gt_offset_tl_per_im[level] for gt_offset_tl_per_im in gt_offset_tl], dim=0)
            )
            hm_tl_weight_list.append(
                torch.cat([hm_tl_weights_per_im[level] for hm_tl_weights_per_im in hm_tl_weights], dim=0)
            )
            offset_tl_weight_list.append(
                torch.cat([offset_tl_weights_im[level] for offset_tl_weights_im in offset_tl_weights], dim=0)
            )
            # br
            gt_hm_br_list.append(
                torch.cat([gt_hm_br_per_im[level] for gt_hm_br_per_im in gt_hm_br], dim=0)
            )
            gt_offset_br_list.append(
                torch.cat([gt_offset_br_per_im[level] for gt_offset_br_per_im in gt_offset_br], dim=0)
            )
            hm_br_weight_list.append(
                torch.cat([hm_br_weights_per_im[level] for hm_br_weights_per_im in hm_br_weights], dim=0)
            )
            offset_br_weight_list.append(
                torch.cat([offset_br_weights_per_im[level] for offset_br_weights_per_im in offset_br_weights], dim=0)
            )
            
        return (gt_hm_tl_list, gt_offset_tl_list, hm_tl_weight_list, offset_tl_weight_list, pos_inds_tl,
                gt_hm_br_list, gt_offset_br_list, hm_br_weight_list, offset_br_weight_list, pos_inds_br)


    def compute_fg_targets(self, targets, padding_shape):
        # targets: [Boxlists]xN
        fg_map_list = []
        fg_weights_list = []
        # B C H W
        # xyxy, xmax - xmin -> W
        for i in range(len(targets)):
            target = targets[i]
            assert target.mode == "xyxy"
            gt_bboxes = target.bbox
            gt_labels = target.get_field('labels')
             # H W
            gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
            fg_map = torch.zeros((self.num_classes, int(padding_shape[0] / 8), int(padding_shape[1] / 8)), dtype=torch.float32)
            fg_weights = torch.zeros((self.num_classes, int(padding_shape[0] / 8), int(padding_shape[1] / 8)), dtype=torch.float32)

            indexs = torch.argsort(gt_areas, descending=True)
            for ind in indexs:
                box = gt_bboxes[ind]
                # H, W
                box_mask = torch.zeros((int(padding_shape[0] / 8), int(padding_shape[1] / 8)), dtype=torch.int64)
                # y, x
                box_mask[int(box[1] / 8):int(box[3] / 8) + 1, int(box[0] / 8):int(box[2] / 8) + 1] = 1
                fg_map[gt_labels[ind]-1][box_mask > 0] = 1
                fg_weights[gt_labels[ind]-1][box_mask > 0] = 1 / gt_areas[ind]

            fg_map_list.append(fg_map)
            fg_weights_list.append(fg_weights)

        # fg_map_list: [C H W]xN -> N C H W
        fg_map_flatten = torch.stack(fg_map_list)
        fg_weights_flatten = torch.stack(fg_weights_list)
        return fg_map_flatten, fg_weights_flatten

    def __call__(self, locations, box_cls,  rou_reg, theta_reg, corners, fg_scores, centerness, targets, padding_shape):
        #

        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1)
        assert num_classes == self.num_classes

        labels, reg_targets = self.prepare_targets(locations, targets)

        box_cls_flatten = []
        centerness_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
      
        for l in range(len(labels)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            labels_flatten.append(labels[l].reshape(-1))
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
            centerness_flatten.append(centerness[l].reshape(-1))

        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        centerness_flatten = torch.cat(centerness_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)

        # polar system
        rou_reg_flatten = []
        theta_reg_flatten = []
        for l in range(len(labels)):
            rou_reg_flatten.append(rou_reg[l].permute(0, 2, 3, 1).reshape(-1, 2))
            theta_reg_flatten.append(theta_reg[l].permute(0, 2, 3, 1).reshape(-1, 2))

        rou_reg_flatten = torch.cat(rou_reg_flatten, dim=0)
        theta_reg_flatten = torch.cat(theta_reg_flatten, dim=0)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)

        rou_reg_flatten = rou_reg_flatten[pos_inds]
        theta_reg_flatten = theta_reg_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]

        num_gpus = get_num_gpus()
        # sync num_pos from all gpus
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

        cls_loss = self.cls_loss_func(
            box_cls_flatten,
            labels_flatten.int()
        ) / num_pos_avg_per_gpu

        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
            _, rou_targets, theta_targets = self.compute_pair_prob_targets(reg_targets_flatten)

            box_regression_flatten = rou_reg_flatten.new_zeros(rou_reg_flatten.shape)

             # l, t, r, b
            rou1, rou2 = rou_reg_flatten.chunk(2, dim=-1)
            sin_theta_1, sin_theta_2 = theta_reg_flatten.chunk(2, dim=-1)
            sin_theta_1 = sin_theta_1.squeeze()
            sin_theta_2 = sin_theta_2.squeeze()
            rou1 = rou1.squeeze()
            rou2 = rou2.squeeze()

            sin_1 = sin_theta_1.sigmoid()
            cos_1 = torch.sqrt(1-sin_1*sin_1)
            sin_2 = sin_theta_2.sigmoid()
            cos_2 = torch.sqrt(1-sin_2*sin_2)
            
            l = rou1 * cos_1
            t = rou1 * sin_1
            r = rou2 * sin_2
            b = rou2 * cos_2

            box_regression_flatten = torch.stack([l, t, r, b], dim=1)

            #
            rou_12 = torch.stack([rou1, rou2],dim=1)
            theta_12 = torch.stack([sin_1, sin_2],dim=1)

            # average sum_centerness_targets from all gpus,
            # which is used to normalize centerness-weighed reg loss
            sum_centerness_targets_avg_per_gpu = \
                reduce_sum(centerness_targets.sum()).item() / float(num_gpus)

            rou_loss = self.rou_loss_func(
                rou_12,
                rou_targets
            ) / num_pos_avg_per_gpu

            rou_loss = 0.0*rou_loss

            theta_loss = self.cos_loss_func(
                theta_12,
                theta_targets
            ) / num_pos_avg_per_gpu

            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            ) / sum_centerness_targets_avg_per_gpu

            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            ) / num_pos_avg_per_gpu
        else:
            reg_loss = box_regression_flatten.sum()
            reduce_sum(centerness_flatten.new_tensor([0.0]))
            centerness_loss = centerness_flatten.sum()
            theta_loss = rou_12.sum()
            rou_loss = theta_12.sum()

        # corner supervision
        corners_tl_score_flatten = []
        corners_br_score_flatten = []
        corners_tl_offset_flatten = []
        corners_br_offset_flatten = []

        (gt_tl_list, gt_offset_tl_list, tl_weights_list, offset_tl_weights_list, pos_inds_tl_list,
         gt_br_list, gt_offset_br_list, br_weights_list, offset_br_weights_list, pos_inds_br_list) = \
        self.prepare_corner_targets(locations, targets)

        num_samples_tl = pos_inds.new_tensor([max(inds.numel(), 1) for inds in pos_inds_tl_list])
        num_samples_br = pos_inds.new_tensor([max(inds.numel(), 1) for inds in pos_inds_br_list])

        num_total_samples_tl = reduce_sum(num_samples_tl.sum()).item()/float(num_gpus)
        num_total_samples_br = reduce_sum(num_samples_br.sum()).item()/float(num_gpus)

        for l in range(len(labels)):
            corners_score = corners[l].permute(0,2,3,1)[:,:,:,:2]
            corners_offset = corners[l].permute(0,2,3,1)[:,:,:,2:]
            corners_tl_score, corners_br_score = corners_score.chunk(2, dim=-1)
            corners_tl_offset, corners_br_offset =corners_offset.chunk(2, dim=-1)
            #
            corners_tl_score_flatten.append(corners_tl_score.reshape(-1).sigmoid())
            corners_br_score_flatten.append(corners_br_score.reshape(-1).sigmoid())
            corners_tl_offset_flatten.append(corners_tl_offset.reshape(-1, 2))
            corners_br_offset_flatten.append(corners_br_offset.reshape(-1, 2))
           
        corners_tl_score_flatten = torch.cat(corners_tl_score_flatten, dim=0)
        corners_br_score_flatten = torch.cat(corners_br_score_flatten, dim=0)
        corners_tl_offset_flatten = torch.cat(corners_tl_offset_flatten, dim=0)
        corners_br_offset_flatten = torch.cat(corners_br_offset_flatten, dim=0)

        # gt
        gt_tl_flatten = torch.cat(gt_tl_list, dim=0)
        gt_offset_tl_flatten = torch.cat(gt_offset_tl_list, dim=0)
        tl_weights_flatten = torch.cat(tl_weights_list, dim=0)
        offset_tl_weights_flatten = torch.cat(offset_tl_weights_list, dim=0)

        gt_br_flatten = torch.cat(gt_br_list, dim=0)
        gt_offset_br_flatten = torch.cat(gt_offset_br_list, dim=0)
        br_weights_flatten = torch.cat(br_weights_list, dim=0)
        offset_br_weights_flatten = torch.cat(offset_br_weights_list, dim=0)
        # loss
        corners_cls_loss = 0
        corners_cls_loss += self.corners_cls_loss_func(
            corners_tl_score_flatten, gt_tl_flatten) / num_total_samples_tl
        corners_cls_loss += self.corners_cls_loss_func(
            corners_br_score_flatten, gt_br_flatten) / num_total_samples_br

        corners_cls_loss /= 2.0

        corners_reg_loss = 0
        corners_reg_loss += self.corners_reg_loss_func(
            corners_tl_offset_flatten,
            gt_offset_tl_flatten, weight=offset_tl_weights_flatten,
            beta=1.0 / 9.0, size_average=False) / num_total_samples_tl
        corners_reg_loss += self.corners_reg_loss_func(
            corners_br_offset_flatten,
            gt_offset_br_flatten, weight=offset_br_weights_flatten,
            beta=1.0 / 9.0, size_average=False) / num_total_samples_br
        corners_reg_loss /= 2.0

        corners_reg_loss = corners_reg_loss

        # fg
        fg_scores_list = []
        gt_fg_scores_list = []
        gt_fg_weights_list = []
        gt_fg_scores, gt_fg_weights = self.compute_fg_targets(targets, padding_shape) # N C H W
        for i in range(len(fg_scores)):
            fg_score = fg_scores[i] # N C H W
            gt_fg_scores_lvl  = F.interpolate(gt_fg_scores, fg_score.shape[-2:]).reshape(-1)
            gt_fg_weights_lvl = F.interpolate(gt_fg_weights, fg_score.shape[-2:]).reshape(-1)
            fg_score = fg_score.reshape(-1)
            #
            fg_scores_list.append(fg_score)
            gt_fg_scores_list.append(gt_fg_scores_lvl)
            gt_fg_weights_list.append(gt_fg_weights_lvl)
        fg_scores_flatten = torch.cat(fg_scores_list)
        gt_fg_scores_flatten = torch.cat(gt_fg_scores_list)
        gt_fg_weights_flatten = torch.cat(gt_fg_weights_list)
        
        total_num_fg = (gt_fg_scores_flatten > 0).sum()
        fg_loss = self.fg_loss_func(
            fg_scores_flatten, 
            gt_fg_scores_flatten, 
            weight=gt_fg_weights_flatten.cuda(),
            avg_factor = total_num_fg
        )
 
        return cls_loss, reg_loss, corners_cls_loss, corners_reg_loss, fg_loss, centerness_loss, rou_loss, theta_loss


def make_fcos_loss_evaluator(cfg):
    loss_evaluator = FCOSLossComputation(cfg)
    return loss_evaluator
