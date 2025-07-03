import torch
import torch.nn as nn
from context.transformer.encoder import TransformerEncoder

class Transformer(nn.Module):
    def __init__(self, d_model = 768, n_heads = 4, n_encoder = 3, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_encoder = n_encoder
        self.encList = nn.ModuleList([
            TransformerEncoder(d_model, n_heads, dropout) for _ in range(n_encoder)
        ])
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        for encoder in self.encList:
            x = encoder(x)
        return x

if __name__ == "__main__":
    x = torch.rand(4, 768, 720)
    model = Transformer()
    out = model(x)
    print(out.shape)