import torch
import torch.nn as nn

class DiversityLoss(nn.Module):
    def __init__(self):
        pass
    
    def forward(self, softmax_probs):
        avg_probs = softmax_probs.mean(dim=(0, 1))  
        
        entropy = - (avg_probs * torch.log(avg_probs + 1e-10)).sum()
        return entropy