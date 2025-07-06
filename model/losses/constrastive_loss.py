import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temp=0.1):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, context, positive, negatives, mask_indices):
        context = context.permute(0, 2, 1)    
        positive = positive.permute(0, 2, 1) 

        B, T, C = context.shape
        K = negatives.shape[2]  

        masked_context = []
        masked_positive = []
        for b in range(B):
            mask = mask_indices[b]  
            masked_context.append(context[b][mask])     
            masked_positive.append(positive[b][mask])  

        max_masked = max(x.size(0) for x in masked_context)
        for i in range(B):
            pad_len = max_masked - masked_context[i].size(0)
            if pad_len > 0:
                pad_tensor = torch.zeros(pad_len, C, device=context.device)
                masked_context[i] = torch.cat([masked_context[i], pad_tensor], dim=0)
                masked_positive[i] = torch.cat([masked_positive[i], pad_tensor], dim=0)

        context = torch.stack(masked_context)     
        positive = torch.stack(masked_positive)   

        pos_sim = self.cos(context, positive)
        context_exp = context.unsqueeze(2)    
        neg_sim = self.cos(context_exp, negatives) 

        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
        logits = logits / self.temp

        labels = torch.zeros(B, context.size(1), dtype=torch.long, device=context.device)


        logits = logits.view(B * context.size(1), K + 1)
        labels = labels.view(-1)

        loss = F.cross_entropy(logits, labels)
        return loss
