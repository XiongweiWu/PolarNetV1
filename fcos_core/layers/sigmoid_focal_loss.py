import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from fcos_core import _C

def separate_sigmoid_focal_loss(pred, target, weight=None, gamma=2.0, alpha=0.25, avg_factor=None):
    #
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)

    pos_inds = target.eq(1)
    neg_inds = target.lt(1)

    pos_weights = weight[pos_inds]

    pos_pred = pred_sigmoid[pos_inds]
    neg_pred = pred_sigmoid[neg_inds]

    pos_loss = -torch.log(pos_pred) * torch.pow(1 - pos_pred, gamma) * pos_weights * alpha
    neg_loss = -torch.log(1 - neg_pred) * torch.pow(neg_pred, gamma) * (1 - alpha)

    if pos_pred.nelement() == 0:
        loss = neg_loss.sum()/avg_factor
    else:
        loss = pos_loss.sum() / pos_weights.sum() + neg_loss.sum()/avg_factor

    return loss

class SEPFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='sum', loss_weight=1.0):
        #
        super(SEPFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None):
        assert self.reduction in (None, 'none', 'mean', 'sum')
        loss_cls = self.loss_weight * separate_sigmoid_focal_loss(
            pred, target, weight, alpha=self.alpha, gamma=self.gamma, avg_factor=avg_factor)
        #
        #if weight is not None:
        #    loss_cls = loss_cls * weight
        if self.reduction == 'sum':
            loss_cls = loss_cls.sum()
        else:
            exit()
        return loss_cls

def gaussian_focal_loss(pred, gaussian_target, alpha=2.0, gamma=4.0):
    eps = 1e-12
    pos_weights = gaussian_target.eq(1)
    neg_weights = (1 - gaussian_target).pow(gamma)
    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    return pos_loss + neg_loss

class GaussianFocalLoss(nn.Module):
    def __init__(self, alpha=2.0, gamma=4.0, reduction='sum', loss_weight=1.0):
        #
        super(GaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, reduction='sum'):
        assert reduction in (None, 'none', 'mean', 'sum')
        loss_reg = self.loss_weight * gaussian_focal_loss(
            pred, target, alpha=self.alpha, gamma=self.gamma)
        if weight is not None:
            loss_reg = loss_reg * weight
        if reduction == 'sum':
            loss_reg = loss_reg.sum()
        else:
            exit()
        return loss_reg

class _SigmoidFocalLoss(Function):
    @staticmethod
    def forward(ctx, logits, targets, gamma, alpha):
        ctx.save_for_backward(logits, targets)
        num_classes = logits.shape[1]
        ctx.num_classes = num_classes
        ctx.gamma = gamma
        ctx.alpha = alpha

        losses = _C.sigmoid_focalloss_forward(
            logits, targets, num_classes, gamma, alpha
        )
        return losses

    @staticmethod
    @once_differentiable
    def backward(ctx, d_loss):
        logits, targets = ctx.saved_tensors
        num_classes = ctx.num_classes
        gamma = ctx.gamma
        alpha = ctx.alpha
        d_loss = d_loss.contiguous()
        d_logits = _C.sigmoid_focalloss_backward(
            logits, targets, d_loss, num_classes, gamma, alpha
        )
        return d_logits, None, None, None, None


sigmoid_focal_loss_cuda = _SigmoidFocalLoss.apply


def sigmoid_focal_loss_cpu(logits, targets, gamma, alpha):
    num_classes = logits.shape[1]
    gamma = gamma[0]
    alpha = alpha[0]
    dtype = targets.dtype
    device = targets.device
    class_range = torch.arange(1, num_classes+1, dtype=dtype, device=device).unsqueeze(0)

    t = targets.unsqueeze(1)
    p = torch.sigmoid(logits)
    term1 = (1 - p) ** gamma * torch.log(p)
    term2 = p ** gamma * torch.log(1 - p)
    return -(t == class_range).float() * term1 * alpha - ((t != class_range) * (t >= 0)).float() * term2 * (1 - alpha)



class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        device = logits.device
        if logits.is_cuda:
            loss_func = sigmoid_focal_loss_cuda
        else:
            loss_func = sigmoid_focal_loss_cpu

        loss = loss_func(logits, targets, self.gamma, self.alpha)
        return loss.sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr
