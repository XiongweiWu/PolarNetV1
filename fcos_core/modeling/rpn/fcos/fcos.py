import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_fcos_postprocessor
from .loss import make_fcos_loss_evaluator

from fcos_core.layers import Scale
from fcos_core.layers import DFConv2d
from fcos_core.layers import smooth_l1_loss
from mmdet.ops import TLPool, BRPool
#from mmcv.cnn import ConvModule


class FCOSHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS
        self.centerness_on_reg = cfg.MODEL.FCOS.CENTERNESS_ON_REG
        self.use_dcn_in_tower = cfg.MODEL.FCOS.USE_DCN_IN_TOWER

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.FCOS.NUM_CONVS):
            if self.use_dcn_in_tower and \
                    i == cfg.MODEL.FCOS.NUM_CONVS - 1:
                conv_func = DFConv2d
            else:
                conv_func = nn.Conv2d

            cls_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels+6, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        # predict rou, 4 directions(tl, br, tr, bl)
        self.rou_pred = nn.Conv2d(
            in_channels+6, 2, kernel_size=3, stride=1,
            padding=1
        )
        # predict sin_theta, 4 directions(tl, br, tr, bl)
        self.theta_pred = nn.Conv2d(
            in_channels+6, 2, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels+6, 1, kernel_size=3, stride=1,
            padding=1
        )
        # learn corner feats
        self.corner_convs = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True)
        )
        #ConvModule(in_channels, in_channels, 3, stride=1, padding=1, norm_cfg=self.norm_cfg)
        #self.corner_convs = nn.Conv2d(
        #    in_channels, in_channels, kernel_size=3, stride=1,
        #    padding=1
        #)

        self.corner_tl = TLPool(in_channels, first_kernel_size=3, kernel_size=1, corner_dim=64)
        self.corner_br = BRPool(in_channels, first_kernel_size=3, kernel_size=1, corner_dim=64)
        self.tl_score_out = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )
        self.br_score_out = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )
        self.tl_offset_out = nn.Conv2d(
            in_channels, 2, kernel_size=3, stride=1,
            padding=1
        )
        self.br_offset_out = nn.Conv2d(
            in_channels, 2, kernel_size=3, stride=1,
            padding=1
        )
        # fg feature learning
        self.fg_score_out = nn.Conv2d(in_channels, num_classes, 1, 1, 0)
        self.fg_embedding = nn.Sequential(
                                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1),
                                nn.GroupNorm(32, in_channels),
                                nn.ReLU(inplace=True)
        )

        #self.fg_embedding = ConvModule(in_channels, in_channels, 1, norm_cfg=norm_cfg)

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, 
                        self.rou_pred, self.theta_pred,
                        self.centerness, self.corner_convs,
                        self.tl_score_out,self.br_score_out,
                        self.tl_offset_out,self.br_offset_out,
                        self.corner_tl, self.corner_br,
                        self.fg_score_out, self.fg_embedding]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias != None:
                        torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)
        torch.nn.init.constant_(self.tl_score_out.bias, bias_value)
        torch.nn.init.constant_(self.br_score_out.bias, bias_value)
        torch.nn.init.constant_(self.fg_score_out.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        rou_reg = []
        theta_reg = []
        centerness = []
        corners = []
        fg_scores = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)
            #
            mid_feat = self.corner_convs(box_tower)
            fg_feat = mid_feat
            cr_feat = mid_feat
            # fg
            fg_score = self.fg_score_out(fg_feat)
            fg_scores.append(fg_score)
            # corner
            fg_feat = self.fg_embedding(fg_feat)
            cls_tower = cls_tower + fg_feat
            box_tower = box_tower + fg_feat
            cr_feat = cr_feat + fg_feat

            tl_feat = self.corner_tl(cr_feat)
            br_feat = self.corner_br(cr_feat)
            tl_score_out = self.tl_score_out(tl_feat)
            tl_offset_out = self.tl_offset_out(tl_feat)
            br_score_out = self.br_score_out(br_feat)
            br_offset_out = self.br_offset_out(br_feat)
            corner_score_out = torch.cat([tl_score_out, br_score_out], dim=1)
            corner_offset_out = torch.cat([tl_offset_out, br_offset_out], dim=1)
            corner_feat = torch.cat([corner_score_out, corner_offset_out], dim=1)
            corners.append(corner_feat)
            # merge feature with original feature map
            cls_tower = torch.cat([cls_tower, corner_feat], dim=1)
            box_tower = torch.cat([box_tower, corner_feat], dim=1)
            

            logits.append(self.cls_logits(cls_tower))
            if self.centerness_on_reg:
                centerness.append(self.centerness(box_tower))
            else:
                centerness.append(self.centerness(cls_tower))

             # rou & theta
            rou_pred = self.scales[l](self.rou_pred(box_tower))
            rou_pred = F.relu(rou_pred)
            if self.training:
                rou_reg.append(rou_pred)
            else:
                rou_reg.append(rou_pred * self.fpn_strides[l])
            #
            theta_pred = self.theta_pred(box_tower)
            # maybe need add some constraint
            #theta_pred = F.relu(theta_pred) #
            theta_reg.append(theta_pred)
            #bbox_reg.append(torch.exp(bbox_pred))

        return logits, rou_reg, theta_reg, corners, fg_scores, centerness


class FCOSModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(FCOSModule, self).__init__()

        head = FCOSHead(cfg, in_channels)

        box_selector_test = make_fcos_postprocessor(cfg)

        loss_evaluator = make_fcos_loss_evaluator(cfg)
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, rou_reg, theta_reg, corners, fg_scores, centerness = self.head(features)
        locations = self.compute_locations(features)
 
        if self.training:
            padding_shape = images.tensors.shape[-2:] # 800 1333, H W
            return self._forward_train(
                locations, box_cls, 
                rou_reg, theta_reg,
                corners, fg_scores,
                centerness, targets, padding_shape
            )
        else:
            return self._forward_test(
                locations, box_cls, rou_reg, theta_reg,
                corners, fg_scores, centerness, images.image_sizes
            )

    def _forward_train(self, locations, box_cls, 
        rou_reg, theta_reg, corners, fg_scores, 
        centerness, targets, padding_shape):
        loss_box_cls, loss_box_reg, loss_corners_cls, loss_corners_reg, loss_fg, loss_centerness, loss_rou, loss_theta = self.loss_evaluator(
            locations, box_cls, rou_reg, theta_reg, corners, fg_scores, centerness, targets, padding_shape
        )
        losses = {
            "loss_cls": loss_box_cls,
            "loss_box": loss_box_reg,
            "loss_corners_cls": loss_corners_cls,
            "loss_corners_reg": loss_corners_reg,
            "loss_fg": loss_fg,
            "loss_centerness": loss_centerness,
            "loss_rou": loss_rou,
            "loss_theta": loss_theta
        }
        return None, losses

    def _forward_test(self, locations, box_cls, 
        rou_reg, theta_reg, corners, fg_scores, 
        centerness, image_sizes):
        boxes = self.box_selector_test(
            locations, box_cls, rou_reg, theta_reg,
            corners, fg_scores, centerness, image_sizes
        )
        return boxes, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

def build_fcos(cfg, in_channels):
    return FCOSModule(cfg, in_channels)
