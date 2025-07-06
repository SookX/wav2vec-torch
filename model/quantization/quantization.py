import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelVectorQuantizer(nn.Module):
    def __init__(self, channels = 512, V = 320, num_codebooks = 2, tau = 1.0):
        super().__init__()
        self.channels = channels
        self.V = V
        self.num_codebooks = num_codebooks
        self.codebook_channel = channels // num_codebooks
        self.tau = tau
        self.codebooks = nn.ModuleList([
            nn.Embedding(V, self.codebook_channel) for _ in range(num_codebooks)
        ])
        self.projections = nn.ModuleList([
            nn.Conv1d(self.codebook_channel, V, kernel_size=1) for _ in range(num_codebooks)
        ])
    
    def forward(self, x):
        batch_size, _, features = x.shape
        x = x.view(batch_size, self.num_codebooks, self.codebook_channel, features)

        stack = []
        probs = []
        for i in range(self.num_codebooks):
            x_i = x[:, i, :, :]
            projections = self.projections[i](x_i)
            logit_projections = F.gumbel_softmax(projections, tau=self.tau, hard=True, dim=1)
            probs.append(logit_projections.permute(0, 2, 1))
            
            embed = self.codebooks[i].weight
            out = torch.matmul(logit_projections.permute(0, 2, 1), embed)

            out = out.permute(0, 2, 1)
            stack.append(out)
        
        quantized = torch.cat(stack, dim=1)
        probs = torch.stack(probs, dim=2)
        return quantized, probs   
        

if __name__ == "__main__":
    x = torch.randn(4, 512, 760)
    model = GumbelVectorQuantizer()
    model(x) 