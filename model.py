import torch.nn as nn

class ECGClassifier(nn.Module):
    def __init__(self):
        super(ECGClassifier, self).__init__()
        # Simple 1D CNN for demonstration
        self.features = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Linear(32, 3) # Labels: Healthy, PAF-Distant, PAF-Preceding

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)