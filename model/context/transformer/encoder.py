import torch
import torch.nn as nn
from transformer.transformer_blocks import AttentionBlock, FeedForwardBlock

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads = 4, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.att_block = AttentionBlock(d_model, n_heads, dropout)
        self.ff_block = FeedForwardBlock(d_model)

    def forward(self, x):
        att = self.att_block(x)
        out = self.ff_block(att)
        return out
    
