import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1    = nn.Conv1d(in_channels, out_channels, kernel_size=7,
                                  stride=stride, padding=3)
        self.bn1      = nn.BatchNorm1d(out_channels)
        self.conv2    = nn.Conv1d(out_channels, out_channels, kernel_size=7,
                                  stride=1, padding=3)
        self.bn2      = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # The "Skip Connection"
        return torch.relu(out)

class PAFClassifier(nn.Module):
    def __init__(self, in_channels=2, num_classes=2):
        super(PAFClassifier, self).__init__()
        
        # Reduced from 64 down to 16 channels
        self.prep = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        
        self.layer1 = ResidualBlock(16, 16)
        self.layer2 = ResidualBlock(16, 32, stride=2) # Downsample
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        self.dropout = nn.Dropout(p=0.5)
        
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        
        x = torch.flatten(x, 1) 
        
        x = self.dropout(x)
        
        logits = self.classifier(x)
        return logits