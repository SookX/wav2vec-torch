import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, in_channels=768, num_groups=16):
        super().__init__()  
        self.in_channels = in_channels
        self.num_groups = num_groups

        self.g_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1, groups=num_groups)

    def forward(self, x):
        pos_emb = self.g_conv(x)
        return x + pos_emb
