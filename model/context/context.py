import torch
import torch.nn as nn


from context.transformer.transformer_model import Transformer
from context.positional_encoding.positional_encoding import PositionalEncoding
from context.projector.feature_projection import FeatureProjection

class Context(nn.Module):
    def __init__(self, in_channels = 512, d_model = 768, num_gropus = 16, n_heads = 4, n_encoder = 3, dropout = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.num_of_groups = num_gropus
        self.n_heads = n_heads
        self.n_encoder = n_encoder
        self.dropout = dropout

        self.fproj_layer = FeatureProjection(in_channels, d_model)
        self.pe_layer = PositionalEncoding(d_model, num_gropus)
        self.transformer = Transformer(d_model, n_heads, n_encoder, dropout)

    def forward(self, x):
        x = self.fproj_layer(x)
        x = self.pe_layer(x)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        return x
    
if __name__ == "__main__":
    x = torch.rand(4, 512, 760)
    model = Context()

    out = model(x)
    print(out.shape)