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
    def __init__(self, in_channels = 512, kernel_size = None, stride=None):
        super().__init__()
        # Chanels - 512 by default
        self.conv1d = nn.Conv1d(in_channels, 512, kernel_size, stride)
        self.layer_norm = nn.LayerNorm(512)
        self.activation = nn.GELU()
    
    def forward(self, x):
        x = self.conv1d(x)             
        x = x.transpose(1, 2)    
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.activation(x)
        return x

class Extractor(nn.Module):
    def __init__(self, number_of_convolutions = 7):
        super().__init__()
        self.number_of_convolutions = number_of_convolutions

    def forward(self, x):
        for i in range(self.number_of_convolutions):
            pass

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.NormalizeAudio = NormalizeAudio()
        self.Extractor = Extractor()
    
    def forward(self, x):
        x = self.NormalizeAudio(x) # 
        x = self.Extractor(x)
    
if __name__ == "__main__":
    x = torch.randn(4, 1, 16000) # (Batch_size, channels, waveform) 
    block = ConvBlock(in_channels=1, kernel_size=10, stride=5)
    out = block(x)
    print(out.shape) 
