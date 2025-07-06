import torch
import torch.nn as nn

class NormalizeAudio(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        mean = x.mean()
        std = x.std()
        return (x - mean) / (std + 1e-8)
   
class ConvBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels, 512, kernel_size, stride, padding)
        self.norm = nn.GroupNorm(1, 512)
        self.activation = nn.GELU()
    
    def forward(self, x):
        x = self.conv1d(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_blocks = nn.ModuleList([
            ConvBlock(1, 10, 5, padding=5),
            ConvBlock(512, 3, 2, padding=1),
            ConvBlock(512, 3, 2, padding=1),
            ConvBlock(512, 3, 2, padding=1),
            ConvBlock(512, 3, 2, padding=1),
            ConvBlock(512, 2, 2, padding=1),
            ConvBlock(512, 2, 2, padding=1),
        ])

    def forward(self, x):
        for i, l in enumerate(self.conv_blocks):
            x = l(x)
        return x
        


class FeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.NormalizeAudio = NormalizeAudio()
        self.Extractor = Extractor()
        self.mask_embedding = nn.Parameter(torch.FloatTensor(512).uniform_())
        self.mask_length = 10
    
    def time_masking(self, x, mask_prob = 0.065):
        batch_size, channels, features = x.shape
        num_masked = int(features * mask_prob / self.mask_length)

        mask_indices = torch.zeros(batch_size, features, dtype=torch.bool, device=x.device)
        for batch in range(batch_size):
            for _ in range(num_masked):
                start = torch.randint(0, features - self.mask_length + 1, (1,)).item()
                mask_indices[batch, start: start + self.mask_length] = True

        transposed_x = x.permute(0, 2, 1)
        transposed_x[mask_indices] = self.mask_embedding
        masked_x = transposed_x.permute(0, 2, 1)

        return masked_x, mask_indices
    
    def get_negatives(self, z, mask_indices, K=100):
        z = z.permute(0, 2, 1) 
        negative_indices = ~mask_indices  
        batch_size, time_steps, feature_dim = z.shape

        all_negatives = []
        max_masked = 0

        for b in range(batch_size):
            masked_pos = torch.nonzero(mask_indices[b], as_tuple=False).squeeze(1)
            max_masked = max(max_masked, masked_pos.numel())

        for b in range(batch_size):
            batch_negatives = []
            masked_pos = torch.nonzero(mask_indices[b], as_tuple=False).squeeze(1)

            for t in masked_pos:
                candidate_indices = torch.nonzero(negative_indices[b], as_tuple=False).squeeze(1)
                if candidate_indices.numel() < K:
                    raise RuntimeError(f"Not enough negative samples for batch {b}, timestep {t}")

                sampled_indices = candidate_indices[torch.randperm(candidate_indices.size(0))[:K]]
                neg_vectors = z[b, sampled_indices] 
                batch_negatives.append(neg_vectors.unsqueeze(0))  

            if len(batch_negatives) < max_masked:
                pad = [torch.zeros_like(batch_negatives[0]) for _ in range(max_masked - len(batch_negatives))]
                batch_negatives.extend(pad)

            batch_negatives = torch.cat(batch_negatives, dim=0) 
            all_negatives.append(batch_negatives.unsqueeze(0))  

        negatives = torch.cat(all_negatives, dim=0)  
        return negatives


            

    def forward(self, x):
        x = self.NormalizeAudio(x) 
        features = self.Extractor(x)     
        masked_features, mask_indices = self.time_masking(features)
        negatives = self.get_negatives(features, mask_indices) 
        return masked_features, features, mask_indices, negatives

    
# Running tests
if __name__ == "__main__":
    x = torch.randn(4, 1, 242720) 
    block = ConvBlock(in_channels=1, kernel_size=10, stride=5, padding=1)
    out = block(x)

    encoder = FeatureEncoder()
    masked_features, features, mask_indices, negatives = encoder(x)
    print(masked_features.shape)