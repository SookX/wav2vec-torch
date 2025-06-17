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
        


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.NormalizeAudio = NormalizeAudio()
        self.Extractor = Extractor()
    
    def forward(self, x):
        x = self.NormalizeAudio(x) 
        x = self.Extractor(x)
        return x
    
# Running tests
if __name__ == "__main__":
    x = torch.randn(4, 1, 16000) 
    block = ConvBlock(in_channels=1, kernel_size=10, stride=5, padding=1)
    out = block(x)

    encoder = Encoder()
    encoder_out = encoder(x)

    print(out.shape) 
    print(encoder_out.shape)