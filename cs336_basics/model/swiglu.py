from torch import nn
import torch

def silu(x: torch.tensor):
    return x * torch.sigmoid(x)

class MySwiGlu(nn.Module):

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self.W3 = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self.W2 = nn.Parameter(torch.empty((d_model, d_ff), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.W1, mean=0.0, std=0.02, a=-0.04, b=0.04)
        nn.init.trunc_normal_(self.W2, mean=0.0, std=0.02, a=-0.04, b=0.04)
        nn.init.trunc_normal_(self.W3, mean=0.0, std=0.02, a=-0.04, b=0.04)

    def forward(self, x):
        output = silu(x @ self.W1.T)
        return (output * (x @ self.W3.T)) @ self.W2.T

