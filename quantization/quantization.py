import torch
import torch.nn as nn

class GumbelVectorQuantizer(nn.Module):
    def __init__(self, channels = 512, V = 320, groups = 2):
        super().__init__()
        self.channels = channels
        self.V = V
        self.groups = groups
        