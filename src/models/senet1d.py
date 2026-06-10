import torch
import torch.nn as nn

class SEBlock1D(nn.Module):
    """
    1D Squeeze-and-Excitation Block for channel-wise feature attention.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock1D, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        # Squeeze: Global average pooling over temporal dimension
        y = x.mean(dim=-1)
        # Excitation: MLP mapping to channel weights
        y = self.fc(y).view(b, c, 1)
        # Scale: channel-wise multiplication
        return x * y.expand_as(x)

class SEResidualBlock(nn.Module):
    """
    1D Residual Block augmented with a Squeeze-and-Excitation attention layer.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, reduction: int = 16):
        super(SEResidualBlock, self).__init__()
        
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
        
        # Squeeze-and-Excitation attention block
        self.se = SEBlock1D(out_channels, reduction=reduction)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)  # Recalibrate channel features
        out += self.shortcut(x)
        return torch.relu(out)

class SEResNetPAFClassifier(nn.Module):
    """
    Squeeze-and-Excitation 1D ResNet for ECG signal classification.
    Input shape: (Batch, 2, Samples)
    Output shape: (Batch, NumClasses)
    """
    def __init__(self, in_channels: int = 2, num_classes: int = 2, reduction: int = 16, hrv_dim: int = 0):
        super(SEResNetPAFClassifier, self).__init__()
        
        # Initial preparation layer
        self.prep = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=31, stride=2, padding=15, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual layers with SE attention
        self.layer1 = nn.Sequential(
            SEResidualBlock(32, 32, reduction=reduction),
            SEResidualBlock(32, 32, reduction=reduction)
        )
        self.layer2 = nn.Sequential(
            SEResidualBlock(32, 64, stride=2, reduction=reduction),
            SEResidualBlock(64, 64, reduction=reduction)
        )
        self.layer3 = nn.Sequential(
            SEResidualBlock(64, 128, stride=2, reduction=reduction),
            SEResidualBlock(128, 128, reduction=reduction)
        )
        self.layer4 = nn.Sequential(
            SEResidualBlock(128, 256, stride=2, reduction=reduction),
            SEResidualBlock(256, 256, reduction=reduction)
        )
        
        # Output layers
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
    model = SEResNetPAFClassifier()
    # Test with 10s window (1280 samples)
    x = torch.randn(8, 2, 1280)
    y = model(x)
    print(f"SEResNet input: {x.shape}, output: {y.shape}")
