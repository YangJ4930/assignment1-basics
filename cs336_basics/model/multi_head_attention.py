import torch.nn as nn
import torch

from cs336_basics.model.linear import MyLinear
from cs336_basics.model.rope import MyRope
from cs336_basics.model.scaled_dot_product_attention import MyScaleDotProductAttention
class MyMultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.q_linear = MyLinear(d_model, d_model)
        self.k_linear = MyLinear(d_model, d_model)
        self.v_linear = MyLinear(d_model, d_model)
        self.o_linear = MyLinear(d_model, d_model)
        self.heads = num_heads
        self.hidden_dim = d_model

    def forward(self, x, rope: MyRope = None, position = None):
        # B L D
        B = x.shape[0]
        L = x.shape[1]
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        Q = Q.view(B, L, self.heads, self.hidden_dim // self.heads).transpose(1, 2)
        K = K.view(B, L, self.heads, self.hidden_dim // self.heads).transpose(1, 2)
        V = V.view(B, L, self.heads, self.hidden_dim // self.heads).transpose(1, 2)
        if rope is not None:
            Q = rope(Q, position.unsqueeze(1))
            K = rope(K, position.unsqueeze(1))
        mask = torch.tril(torch.ones(L, L)).bool()
        scaleDotProductAttention = MyScaleDotProductAttention()
        out = scaleDotProductAttention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(B, L, self.hidden_dim)
        return self.o_linear(out)
