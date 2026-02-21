import torch.nn as nn

from cs336_basics.model.multi_head_attention import MyMultiHeadAttention
from cs336_basics.model.rms_norm import MyRmsNorm
from cs336_basics.model.swiglu import MySwiGlu


class MyTransformerBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, device=None, dtype=None):
        super().__init__()
        self.rmsNorm1 = MyRmsNorm(d_model=d_model, device=device, dtype=dtype)
        self.rmsNorm2 = MyRmsNorm(d_model=d_model, device=device, dtype=dtype)
        self.mha = MyMultiHeadAttention(d_model, num_heads, device=device, dtype=dtype)
        self.position_feed_forward_network = MySwiGlu(d_model, d_ff, device, dtype)

    def forward(self, in_features, rope):
        x = in_features + self.mha(self.rmsNorm1(in_features), rope)
        x = x + self.position_feed_forward_network(self.rmsNorm2(x))
        return x

