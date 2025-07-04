import torch
import torch.nn as nn
from context.context import Context
from extractor.extractor import FeatureEncoder
from quantization.quantization import GumbelVectorQuantizer

class Wav2Vec(nn.Module):
    def __init__(self, d_model = 768):
        super().__init__()
        self.feature_enc = FeatureEncoder()
        self.quantizer = GumbelVectorQuantizer()
        self.context = Context(d_model=d_model)
        self.out_proj_layer = nn.Conv1d(d_model, 512, 1) # Only for pre-training

    def forward(self, x):
        x, mask_indices = self.feature_enc(x)
        targets = self.quantizer(x)
        preds = self.context(x)
        preds = self.out_proj_layer(preds)

        return preds, targets, mask_indices


if __name__ == "__main__":
    x = torch.rand(10, 1, 242720)
    model = Wav2Vec()
    preds, targets, mask_indices = model(x)
    print(preds.shape)
    print(targets.shape)
    print(mask_indices.shape)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {pytorch_total_params}")