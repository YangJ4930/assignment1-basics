import torch.nn as nn
import torch

class MyRope(nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        freq_base = torch.arange(0, d_k, 2, device=device)
        freq_base = 1.0 / (theta ** (freq_base / d_k))
        position = torch.arange(0, max_seq_len, device=device)
        angles = torch.outer(position, freq_base)
        self.register_buffer("cos", torch.cos(angles))
        self.register_buffer("sin", torch.sin(angles))


    def forward(self, in_query_or_key, token_positions):
        """
           Run RoPE for a given input tensor.

           Args:
               d_k (int): Embedding dimension size for the query or key tensor.
               theta (float): RoPE parameter.
               max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
               in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
               token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
           Returns:
               Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
           """
        cos = self.cos[token_positions, :]
        sin = self.sin[token_positions, :]
        x1 = in_query_or_key[..., 0::2]
        x2 = in_query_or_key[..., 1::2]
        rotate1 = x1 * cos - x2 * sin
        rotate2 = x1 * sin + x2 * cos
        res = torch.zeros_like(in_query_or_key)
        res[..., 0::2] = rotate1
        res[..., 1::2] = rotate2
        return res

