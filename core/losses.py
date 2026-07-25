import torch
from torch import nn
from torch.nn import functional as F

class ReconstructionLoss(nn.Module):
    def __init__(self, loss_type='L2'):
        super().__init__()
        self.loss_fn = F.mse_loss if loss_type == 'L2' else F.l1_loss

    def forward(self, predict, target):
        fg_mask = target >= 0
        predict, target = predict[fg_mask], target[fg_mask]

        # return (torch.abs(predict - target) * self.loss_fn(predict, target)).mean()
        return self.loss_fn(predict, target).mean()
    
class BalancedReconstructionLoss(nn.Module):
    def __init__(self, loss_type='L2'):
        super().__init__()
        self.loss_fn = F.mse_loss if loss_type == 'L2' else F.l1_loss

    def forward(self, pos_predict, pos_target, neg_predict, neg_target):
        fg_mask = pos_target >= 0
        pos_predict, pos_target = pos_predict[fg_mask], pos_target[fg_mask]
        pos_loss = self.loss_fn(pos_predict, pos_target).mean()

        if neg_predict.shape[0] > 0:
            fg_mask = neg_target >= 0
            neg_predict, neg_target = neg_predict[fg_mask], neg_target[fg_mask]
            neg_loss = self.loss_fn(neg_predict, neg_target).mean()

            loss = (pos_loss + neg_loss) / 2.
        else:
            loss = pos_loss

        return loss
