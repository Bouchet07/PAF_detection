import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    1D Residual Block with skip connections and batch normalization.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=15,
            stride=stride, padding=7, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=15,
            stride=1, padding=7, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)

class PAFClassifier(nn.Module):
    """
    1D ResNet classifier for paroxysmal atrial fibrillation imminence prediction.
    Input shape: (Batch, 2, Samples)
    Output shape: (Batch, NumClasses)
    """
    def __init__(self, in_channels: int = 2, num_classes: int = 2, hrv_dim: int = 0):
        super(PAFClassifier, self).__init__()
        
        # Initial preparation layer
        self.prep = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=31, stride=2, padding=15, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual layers
        self.layer1 = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 32)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128)
        )
        self.layer4 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256)
        )
        
        # Output pool and linear projection
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.5)
        
        self.hrv_dim = hrv_dim
        if hrv_dim > 0:
            self.classifier = nn.Linear(256 + hrv_dim, num_classes)
        else:
            self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor, hrv: torch.Tensor = None) -> torch.Tensor:
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1) 
        x = self.dropout(x)
        
        if self.hrv_dim > 0 and hrv is not None:
            x = torch.cat([x, hrv], dim=1)
            
        logits = self.classifier(x)
        return logits

if __name__ == "__main__":
    model = PAFClassifier()
    # Test with 10s window at 128Hz (1280 samples)
    x = torch.randn(8, 2, 1280)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    
    # Test with 30s window (3840 samples)
    x = torch.randn(8, 2, 3840)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
