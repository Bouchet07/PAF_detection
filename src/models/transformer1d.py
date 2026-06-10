import torch
import torch.nn as nn
import math

class PositionalEncoding1D(nn.Module):
    """
    Learnable 1D Positional Encodings added to the token sequence.
    """
    def __init__(self, d_model: int, max_len: int = 2000):
        super(PositionalEncoding1D, self).__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        # Initialize with standard normal
        nn.init.normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, SeqLen, Channels)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class CNNTransformerPAFClassifier(nn.Module):
    """
    Hybrid CNN-Transformer network for ECG signal classification.
    Uses 1D Convolutions for feature extraction and temporal downsampling,
    followed by a Transformer Encoder to capture long-range beat-to-beat correlations.
    Input shape: (Batch, 2, Samples)
    Output shape: (Batch, NumClasses)
    """
    def __init__(
        self, 
        in_channels: int = 2, 
        num_classes: int = 2, 
        d_model: int = 128, 
        nhead: int = 4, 
        num_encoder_layers: int = 3, 
        dim_feedforward: int = 256, 
        dropout: float = 0.2,
        hrv_dim: int = 0
    ):
        super(CNNTransformerPAFClassifier, self).__init__()
        
        # 1. CNN Frontend (downsamples length by a factor of 8)
        self.frontend = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout1d(p=0.1),
            
            nn.Conv1d(32, 64, kernel_size=11, stride=2, padding=5, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout1d(p=0.1),
            
            nn.Conv1d(64, d_model, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding1D(d_model=d_model, max_len=2000)
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        # 4. Classifier Head
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        self.hrv_dim = hrv_dim
        if hrv_dim > 0:
            self.classifier = nn.Sequential(
                nn.Linear(d_model + hrv_dim, d_model),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(d_model, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(d_model, num_classes)
            )
        
        self.d_model = d_model

    def forward(self, x: torch.Tensor, hrv: torch.Tensor = None) -> torch.Tensor:
        # Input shape: (Batch, 2, Samples)
        
        # 1. Feature extraction & downsampling
        features = self.frontend(x)  # Shape: (Batch, d_model, Samples // 8)
        
        # 2. Reshape for Transformer: (Batch, SeqLen, d_model)
        x_trans = features.transpose(1, 2)
        
        # 3. Add Positional Encoding
        x_trans = self.pos_encoder(x_trans)
        
        # 4. Transformer Attention block
        attention_out = self.transformer_encoder(x_trans)  # Shape: (Batch, SeqLen, d_model)
        
        # 5. Pooling & Linear Projection
        pooled = attention_out.transpose(1, 2)  # Shape: (Batch, d_model, SeqLen)
        pooled = self.avgpool(pooled)           # Shape: (Batch, d_model, 1)
        pooled = torch.flatten(pooled, 1)       # Shape: (Batch, d_model)
        
        if self.hrv_dim > 0 and hrv is not None:
            pooled = torch.cat([pooled, hrv], dim=1)
            
        logits = self.classifier(pooled)
        return logits

if __name__ == "__main__":
    model = CNNTransformerPAFClassifier()
    # Test with 10s window (1280 samples)
    x = torch.randn(8, 2, 1280)
    y = model(x)
    print(f"CNN-Transformer input: {x.shape}, output: {y.shape}")
