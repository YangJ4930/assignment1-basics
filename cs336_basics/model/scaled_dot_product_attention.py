import math

import torch.nn as nn
import torch

from cs336_basics.model.softmax import MySoftmax


class MyScaleDotProductAttention(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, mask):
        dim = q.shape[-1]
        softmax = MySoftmax()
        scores = q @ k.transpose(-1, -2) / math.sqrt(dim)
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        score = softmax(scores)
        return score @ v
