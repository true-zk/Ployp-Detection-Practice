"""
@Description: Metrics, copied from @Ref
@Ref: https://github.com/rishikksh20/ResUnet/blob/master/utils/metrics.py
@Author: Ken Zh0ng
@date: 2024-06-08
"""
import torch.nn as nn

class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_mean=True) -> None:
        super().__init__()
        
    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        gd = target.contiguous().view(-1)
        
        # BCE Loss
        BCE_loss = nn.BCELoss()(pred, gd).double()
        
        # Dice loss
        Dice_coef = (2. * (pred * gd).double().sum() + 1) / (
                    pred.double().sum() + gd.double().sum() + 1)
        
        return BCE_loss + (1 - Dice_coef)


# https://github.com/pytorch/examples/blob/master/imagenet/main.py
class MetricTracker(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

# https://stackoverflow.com/questions/48260415/pytorch-how-to-compute-iou-jaccard-index-for-semantic-segmentation
def jaccard_index(input, target):

    intersection = (input * target).long().sum().data.cpu()[0]
    union = (
        input.long().sum().data.cpu()[0]
        + target.long().sum().data.cpu()[0]
        - intersection
    )

    if union == 0:
        return float("nan")
    else:
        return float(intersection) / float(max(union, 1))


# https://github.com/pytorch/pytorch/issues/1249
def dice_coeff(input, target):
    num_in_target = input.size(0)

    smooth = 1.0

    pred = input.view(num_in_target, -1)
    truth = target.view(num_in_target, -1)

    intersection = (pred * truth).sum(1)

    loss = (2.0 * intersection + smooth) / (pred.sum(1) + truth.sum(1) + smooth)

    return loss.mean().item()