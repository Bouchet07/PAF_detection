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
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, kernel_size: int = 15, reduction: int = 16):
        super(SEResidualBlock, self).__init__()
        
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel_size,
            stride=1, padding=padding, bias=False
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
    def __init__(
        self, 
        in_channels: int = 2, 
        num_classes: int = 2, 
        reduction: int = 16, 
        hrv_dim: int = 0,
        layers: list = [2, 2, 2, 2],
        channels: list = [32, 64, 128, 256],
        kernel_size: int = 15,
        dropout: float = 0.5
    ):
        super(SEResNetPAFClassifier, self).__init__()
        
        # Initial preparation layer
        init_channels = channels[0]
        self.prep = nn.Sequential(
            nn.Conv1d(in_channels, init_channels, kernel_size=31, stride=2, padding=15, bias=False),
            nn.BatchNorm1d(init_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Build list of sequential blocks
        layers_list = []
        current_channels = init_channels
        for i, num_blocks in enumerate(layers):
            layer_blocks = []
            out_channels = channels[i]
            stride = 2 if i > 0 else 1
            layer_blocks.append(SEResidualBlock(current_channels, out_channels, stride=stride, kernel_size=kernel_size, reduction=reduction))
            current_channels = out_channels
            for _ in range(1, num_blocks):
                layer_blocks.append(SEResidualBlock(current_channels, current_channels, stride=1, kernel_size=kernel_size, reduction=reduction))
            layers_list.append(nn.Sequential(*layer_blocks))
            
        # Register layers as named attributes to keep state_dict compatible
        for idx, layer_module in enumerate(layers_list):
            setattr(self, f"layer{idx+1}", layer_module)
            
        self.num_layers = len(layers_list)
        
        # Output layers
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout)
        
        self.hrv_dim = hrv_dim
        final_channels = channels[-1]
        if hrv_dim > 0:
            self.classifier = nn.Linear(final_channels + hrv_dim, num_classes)
        else:
            self.classifier = nn.Linear(final_channels, num_classes)

    def forward(self, x: torch.Tensor, hrv: torch.Tensor = None) -> torch.Tensor:
        x = self.prep(x)
        for idx in range(self.num_layers):
            layer = getattr(self, f"layer{idx+1}")
            x = layer(x)
        
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
