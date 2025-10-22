import torch, torch.nn as nn

class VEPHead(nn.Module):
    def __init__(self, d_model: int, hidden: int = None, p: float = 0.1):
        super().__init__()
        h = hidden or (d_model // 2)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, h),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(h, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
