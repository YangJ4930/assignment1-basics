import torch.nn as nn
import torch

class MyCrossEntropy(nn.Module):

    def __init__(self):
        super().__init__()

    def log_exp_sum(self, x):
        x_max = x.max(dim=-1, keepdim=True).values
        return x - x_max - torch.log(torch.exp(x - x_max).sum(dim=-1, keepdim=True))

    def forward(self, inputs, targets):
        """Given a tensor of inputs and targets, compute the average cross-entropy
        loss across examples.

        Args:
            inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
                unnormalized logit of jth class for the ith example.
            targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
                Each value must be between 0 and `num_classes - 1`.

        Returns:
            Float[Tensor, ""]: The average cross-entropy loss across examples.
        """
        batch_size = inputs.shape[0]
        range = torch.arange(0, batch_size, 1)
        logits = self.log_exp_sum(inputs)
        x = logits[range, targets]

        return -1 * x.mean(dim=-1)
