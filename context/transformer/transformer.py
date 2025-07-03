import torch
import torch.nn as nn
from positional_encoding import PositionalEncoding

class Transformer(nn.Module):
    def __init__(self, in_channels = 768):
        super().__init__()
        self.pos_enc = PositionalEncoding(in_channels)
    
    def forward(self, x):
        x = self.pos_enc(x)
        return x

if __name__ == "__main__":
    x = torch.rand(4, 768, 720)
    model = Transformer()
    out = model(x)
    print(out.shape)