import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temp=0.1):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)
    
    def forward(self, context, positive, negatives):

        B, T, C = context.shape
        K = negatives.shape[2]

        pos_sim = self.cos(context, positive)                  
        context_exp = context.unsqueeze(2)                 
        neg_sim = self.cos(context_exp, negatives)            

        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1) 
        logits = logits / self.temp

        labels = torch.zeros(B, T, dtype=torch.long, device=context.device)
        logits = logits.view(B * T, K + 1)
        labels = labels.view(B * T)

        loss = F.cross_entropy(logits, labels)
        return loss
