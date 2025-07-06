import torch
import torch.nn as nn
from model.losses.constrastive_loss import ContrastiveLoss
from model.losses.diversity_loss import DiversityLoss

class Loss(nn.Module):
    def __init__(self, alpha = 0.2, temp = 0.1):
        super().__init__()
        self.alpha = alpha
        self.temp = temp
        self.cl = ContrastiveLoss(temp)
        self.dl = DiversityLoss()
    
    def forward(self, context, positive, negatives, softmax_probs, mask_indices):
        loss = self.cl(context, positive, negatives, mask_indices) + self.alpha * self.dl(softmax_probs)
        return loss