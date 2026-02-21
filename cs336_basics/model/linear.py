import torch.nn as nn
import torch

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        # Initialize weights with trunc_normal_ as requested
        # Using standard deviation 0.02 (common for LLMs) and truncating at 2*std
        nn.init.trunc_normal_(self.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

