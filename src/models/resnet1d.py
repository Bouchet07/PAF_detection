import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    1D Residual Block with skip connections and batch normalization.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, kernel_size: int = 15):
        super(ResidualBlock, self).__init__()
        
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
    def __init__(
        self, 
        in_channels: int = 2, 
        num_classes: int = 2, 
        hrv_dim: int = 0,
        layers: list = [2, 2, 2, 2],
        channels: list = [32, 64, 128, 256],
        kernel_size: int = 15,
        dropout: float = 0.5
    ):
        super(PAFClassifier, self).__init__()
        
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
            layer_blocks.append(ResidualBlock(current_channels, out_channels, stride=stride, kernel_size=kernel_size))
            current_channels = out_channels
            for _ in range(1, num_blocks):
                layer_blocks.append(ResidualBlock(current_channels, current_channels, stride=1, kernel_size=kernel_size))
            layers_list.append(nn.Sequential(*layer_blocks))
            
        # Register layers as named attributes to keep state_dict compatible
        for idx, layer_module in enumerate(layers_list):
            setattr(self, f"layer{idx+1}", layer_module)
            
        self.num_layers = len(layers_list)
        
        # Output pool and linear projection
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
    model = PAFClassifier()
    # Test with 10s window at 128Hz (1280 samples)
    x = torch.randn(8, 2, 1280)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    
    # Test with 30s window (3840 samples)
    x = torch.randn(8, 2, 3840)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
