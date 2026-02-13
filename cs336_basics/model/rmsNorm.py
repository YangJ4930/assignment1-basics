import math

import torch.nn as nn
import torch


class MyRmsNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # batch_size seq_len dim
        # 求平方
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        result = x * rms * self.weight
        return result.to(in_dtype)