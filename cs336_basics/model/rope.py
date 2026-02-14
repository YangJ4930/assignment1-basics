import torch.nn as nn
import torch

class MyRope(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, dk, theta, max_seq_len, in_query_or_key, token_positions):
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
        freq_base = torch.arange(0, dk, 2, device=in_query_or_key.device, dtype=in_query_or_key.dtype)
        freq_base = 1.0 / (theta ** (freq_base / dk))
        angles = token_positions.to(in_query_or_key.dtype).unsqueeze(-1) * freq_base
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        x1 = in_query_or_key[..., 0::2]
        x2 = in_query_or_key[..., 1::2]
        rotate1 = x1 * cos - x2 * sin
        rotate2 = x1 * sin + x2 * cos
        res = torch.zeros_like(in_query_or_key)
        res[..., 0::2] = rotate1
        res[..., 1::2] = rotate2
        return res

