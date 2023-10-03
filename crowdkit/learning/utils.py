__all__ = [
    "differentiable_ds",
    "batch_identity_matrices",
]

from typing import Optional

import torch
import torch.nn.functional as F


def differentiable_ds(
    outputs: torch.Tensor, confusion_matrices: torch.Tensor
) -> torch.Tensor:
    """
    Differentiable Dawid-Skene logit transformation.
    Args:
        outputs (torch.Tensor): Tensor of shape (batch_size, input_dim)
        confusion_matrices (torch.Tensor): Tensor of shape (batch_size, input_dim, input_dim)

    Returns:
        Tensor of shape (batch_size, input_dim)
    """
    normalized_matrices = F.softmax(confusion_matrices, dim=-1).transpose(2, 1)
    return torch.log(
        torch.einsum(
            "lij,ljk->lik",
            normalized_matrices,
            F.softmax(outputs, dim=-1).unsqueeze(-1),
        ).squeeze()
    )


def batch_identity_matrices(
    batch_size: int,
    dim_size: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Creates a batch of identity matrices.
    Args:
        batch_size (int): Batch size.
        dim_size (int): Dimension size.
        device (torch.device): Device to place the matrices on.
        dtype (torch.dtype): Data type of the matrices.

    Returns:
        Tensor of shape (batch_size, dim_size, dim_size)
    """
    x = torch.eye(dim_size, dtype=dtype, device=device)
    x = x.reshape((1, dim_size, dim_size))
    return x.repeat(batch_size, 1, 1)
