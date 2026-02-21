from torch import nn
import torch

from cs336_basics.model.linear import MyLinear


def silu(x: torch.tensor):
    return x * torch.sigmoid(x)

class MySwiGlu(nn.Module):

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        # self.W1 = nn.Parameter(torch.empty((d_model, d_ff), device=device, dtype=dtype))
        # self.W3 = nn.Parameter(torch.empty((d_model, d_ff), device=device, dtype=dtype))
        # self.W2 = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self.W1 = MyLinear(d_model, d_ff, device, dtype)
        self.W2 = MyLinear(d_ff, d_model, device, dtype)
        self.W3 = MyLinear(d_model, d_ff, device, dtype)

    def forward(self, x):
        output = silu(self.W1(x))
        return self.W2(output * (self.W3(x)))

