import torch.nn as nn
import torch


class MySoftmax(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_max = x.max()
        x = x - x_max
        x_exp = torch.exp(x)
        x_exp_sum = x_exp.sum(dim=-1, keepdim=True)
        return torch.exp(x) / x_exp_sum
