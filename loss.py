import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# 完全信任标签, 不予转换, SPLC不使用, Hill不使用, 使用Focal Margin Loss + ASL组合
# 正标签1: Focal Margin Loss(放宽难样本限制的挖掘)
# 负标签0: ASL(剔除容易样本0.00~0.05, 负样本支配时使用, 自动剔除一些无价值负样本)
# 标签平滑
class ASLMarginLossSmooth(nn.Module):
    r"""
    Loss+使用Focal Margin Loss,
    Loss-使用ASL,
    加ignore_index, class_weights
    Args:
        margin (float): Margin value. Default: 1
        gamma (float): Hard mining value. Default: 2
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'sum'``
        """

    def __init__(self,
                 margin: float = 1.0,
                 gamma_neg: float = 4.0, # 负样本缩的更厉害
                 gamma_pos: float = 2.0,
                 eps: float = 1e-8,
                 clip: float = 0.05,
                 ignore_index: int = 255,
                 class_weights: list = None,
                 reduction: str = 'sum',
                 smooth: float = 0.0,
                 **kwargs) -> None:
        super().__init__()
        self.margin = margin
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.eps = eps
        self.clip = clip
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.class_weights = class_weights
        self.smooth = smooth

    def reset_class_weights(self,new_value):
        self.class_weights = new_value


    def forward(self, logits: torch.Tensor, targets: torch.LongTensor) -> torch.Tensor:
        """
        call function as forward
        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`
        Returns:
            torch.Tensor: loss
        """
        # Subtract margin for positive logits
        # 把正标签对应位置的预测值减去margin, 负标签不变
        logits = torch.where(targets == 1, logits - self.margin, logits)

        pred = torch.sigmoid(logits)
        # Define epsilon so that the backpropagation will not result in NaN
        pred = pred.clamp(min=self.eps, max=1 - self.eps)

        pred_pos = pred
        pred_neg = 1 - pred

        # focal weights calculate
        pt = pred_pos * targets + pred_neg * (1 - targets)
        one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        focal_weight = torch.pow(1 - pt, one_sided_gamma)

        if self.smooth:
            targets = targets * (1 - self.smooth) + (1 - targets) * self.smooth

        # Asymmetric Clipping
        # 剔除预测概率为0.00~0.05的容易负样本
        if self.clip is not None and self.clip > 0:
            pred_neg = (pred_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = targets * torch.log(pred_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(pred_neg.clamp(min=self.eps))
        loss = - (los_pos + los_neg)

        loss *= focal_weight

        # loss+(Focal Margin Loss) = (1 - sigmoid(logits-margin))^gamma_pos * -log(sigmoid(logits-margin))
        #                          = - (1-pred)^2 * log(pred)
        #   { pred = sigmoid(logits-margin) }
        # margin 将focal loss的峰值梯度拓宽, 从约0~0.2的峰值变为约0~0.5的峰值, 这样预测概率为0.3~0.5的半难样本也能得到较大的梯度

        # loss-(ASL) = - (1 - pred_neg)^4 * log(pred_neg)
        #            = - (1 - min(1-pred+0.05, 1))^4 * log(min(1-pred+0.05, 1))
        #            = 0 (if 0.00 < pred < 0.05)
        #            = - (pred-0.05)^4 * log(1-(pred-0.05)) (if 0.05 < pred < 1.00)
        #   { pred = sigmoid(logits) - 0.05 }
        #            = - pred^4 * log(1-pred) (if 0.05 < pred < 1.00)
        # 0.00~0.05的负样本loss截断为0, 0.05~1.00的负样本当做0.00~0.95计算, 相当于loss整体都降低一点
        # gamma_neg=4让负样本缩的更厉害


        loss *= (targets != self.ignore_index)
        # class weights
        if self.class_weights:
            loss *= torch.FloatTensor(self.class_weights).to(loss.device)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss



class MultiFocalLoss(nn.Module):
    """
    Focal_Loss= -1*alpha*((1-pt)**gamma)*log(pt)
    Args:
        num_class: number of classes
        alpha: class balance factor shape=[num_class, ]
        gamma: hyper-parameter
        reduction: reduction type
    """

    def __init__(self, num_class, alpha=None, gamma=2, reduction='mean'):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = 1e-4
        self.alpha = alpha
        if alpha is None:
            self.alpha = torch.ones(num_class, ) - 0.5
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.as_tensor([alpha] * num_class)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.as_tensor(alpha)
        if self.alpha.shape[0] != num_class:
            raise RuntimeError('the length not equal to number of class')

    def forward(self, logit, target):
        # assert isinstance(self.alpha,torch.Tensor)\
        alpha = self.alpha.to(logit.device)
        prob = F.softmax(logit, dim=1)

        if prob.dim() > 2:
            # used for 3d-conv:  N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            N, C = logit.shape[:2]
            prob = prob.view(N, C, -1)
            prob = prob.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            prob = prob.view(-1, prob.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]

        ori_shp = target.shape
        target = target.view(-1, 1)

        prob = prob.gather(1, target).view(-1) + self.smooth  # avoid nan
        logpt = torch.log(prob)
        # alpha_class = alpha.gather(0, target.squeeze(-1))
        alpha_weight = alpha[target.squeeze().long()]
        loss = -alpha_weight * torch.pow(torch.sub(1.0, prob), self.gamma) * logpt

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'none':
            loss = loss.view(ori_shp)

        return loss