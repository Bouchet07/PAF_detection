import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaSimpleBlock(nn.Module):
    """
    A simplified, pure-PyTorch implementation of the Mamba (Selective SSM) block.
    Does not require custom CUDA extensions, runs out-of-the-box on GPU and CPU.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # 1D Convolution along sequence dimension
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1
        )
        
        # Selective SSM parameter projections: B (d_state), C (d_state), and dt_input (d_inner)
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + self.d_inner, bias=False)
        
        # Delta parameter projection (dt)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # Initialize A parameter: learned matrix log(A) for stability
        # A shape: (d_inner, d_state)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D parameter (skip connection parameter for SSM)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Out projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x shape: (Batch, Seq_len, d_model)
        batch, seq_len, _ = x.shape
        
        # 1. Input projection: split into branches
        xz = self.in_proj(x) # (B, L, 2 * d_inner)
        x_branch, z_branch = xz.chunk(2, dim=-1) # (B, L, d_inner)
        
        # 2. Convolution branch
        x_conv = x_branch.transpose(1, 2) # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len] # Keep sequence length consistent
        x_conv = x_conv.transpose(1, 2) # (B, L, d_inner)
        x_conv = F.silu(x_conv)
        
        # 3. Selective SSM
        x_proj_out = self.x_proj(x_conv)
        dt_input, B, C = torch.split(x_proj_out, [self.d_inner, self.d_state, self.d_state], dim=-1)
        
        # Discretization delta (dt)
        dt = F.softplus(self.dt_proj(dt_input)) # (B, L, d_inner)
        
        # Compute discrete A and B
        A = -torch.exp(self.A_log) # (d_inner, d_state)
        
        # Selective Scan: recurrent update in PyTorch
        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        y_ssm = []
        
        # Precompute dt * A (shape: B, L, d_inner, d_state)
        dt_A = torch.einsum("bli,in->blin", dt, A)
        dA = torch.exp(dt_A) # (B, L, d_inner, d_state)
        
        # Precompute dt * B (shape: B, L, d_inner, d_state)
        dB = torch.einsum("bli,bln->blin", dt, B) # dt is (B, L, d_inner), B is (B, L, d_state)
        
        # Loop for selective scan
        for t in range(seq_len):
            x_t = x_conv[:, t, :] # (B, d_inner)
            dA_t = dA[:, t, :, :] # (B, d_inner, d_state)
            dB_t = dB[:, t, :, :] # (B, d_inner, d_state)
            C_t = C[:, t, :] # (B, d_state)
            
            # Update state h
            h = dA_t * h + dB_t * x_t.unsqueeze(-1) # (B, d_inner, d_state)
            
            # Compute output
            y_t = torch.einsum("bn,bin->bi", C_t, h) # (B, d_inner)
            y_ssm.append(y_t)
            
        y_ssm = torch.stack(y_ssm, dim=1) # (B, L, d_inner)
        
        # Add skip connection from SSM skip D
        y_ssm = y_ssm + x_conv * self.D
        
        # 4. Gating branch with z_branch
        y_gated = y_ssm * F.silu(z_branch)
        
        # 5. Out projection
        out = self.out_proj(y_gated)
        return out

class MambaPAFClassifier(nn.Module):
    """
    CNN-Mamba Hybrid network for ECG signal classification.
    Uses 1D Convolutions for feature extraction and temporal downsampling,
    followed by stacked Mamba blocks to capture long-range temporal correlations.
    Input shape: (Batch, 2, Samples)
    Output shape: (Batch, NumClasses)
    """
    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 2,
        d_model: int = 64,
        num_layers: int = 3,
        d_state: int = 16,
        dropout: float = 0.2,
        hrv_dim: int = 0
    ):
        super().__init__()
        
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
        
        # 2. Mamba Encoder Stack (no Positional Encoding needed!)
        self.mamba_layers = nn.ModuleList([
            MambaSimpleBlock(d_model=d_model, d_state=d_state, expand=2)
            for _ in range(num_layers)
        ])
        
        # 3. Classifier Head
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

    def forward(self, x: torch.Tensor, hrv: torch.Tensor = None) -> torch.Tensor:
        # Input shape: (Batch, 2, Samples)
        
        # 1. Feature extraction & downsampling
        features = self.frontend(x)  # Shape: (Batch, d_model, Samples // 8)
        
        # 2. Reshape for Mamba: (Batch, SeqLen, d_model)
        x_mamba = features.transpose(1, 2)
        
        # 3. Pass through Mamba block stack
        for layer in self.mamba_layers:
            x_mamba = layer(x_mamba)
            
        # 4. Pooling & Linear Projection
        pooled = x_mamba.transpose(1, 2)  # Shape: (Batch, d_model, SeqLen)
        pooled = self.avgpool(pooled)           # Shape: (Batch, d_model, 1)
        pooled = torch.flatten(pooled, 1)       # Shape: (Batch, d_model)
        
        if self.hrv_dim > 0 and hrv is not None:
            pooled = torch.cat([pooled, hrv], dim=1)
            
        logits = self.classifier(pooled)
        return logits

if __name__ == "__main__":
    model = MambaPAFClassifier()
    # Test with 10s window (1280 samples)
    x = torch.randn(8, 2, 1280)
    y = model(x)
    print(f"CNN-Mamba input: {x.shape}, output: {y.shape}")
