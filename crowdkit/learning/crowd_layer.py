__all__ = [
    "CrowdLayer",
]

from typing import Optional

import torch
from torch import nn

from crowdkit.learning.utils import batch_identity_matrices


def crowd_layer_mw(
    outputs: torch.Tensor, workers: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """
    CrowdLayer MW transformation. Defined by multiplication on squared confusion matrix.
    This complies with the Dawid-Skene model.

    Args:
        outputs (torch.Tensor): Tensor of shape (batch_size, input_dim)
        workers (torch.Tensor): Tensor of shape (batch_size,) containing the worker IDs.
        weight (torch.Tensor): Tensor of shape (batch_size, 1) containing the workers' confusion matrices.

    Returns:
        torch.Tensor: Tensor of shape (batch_size, input_dim)
    """
    return torch.einsum(
        "lij,ljk->lik", weight[workers], outputs.unsqueeze(-1)
    ).squeeze()


def crowd_layer_vw(
    outputs: torch.Tensor, workers: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """
    CrowdLayer VW transformation. A linear transformation of the input without the bias.

    Args:
        outputs (torch.Tensor): Tensor of shape (batch_size, input_dim)
        workers (torch.Tensor): Tensor of shape (batch_size,) containing the worker IDs.
        weight (torch.Tensor): Tensor of shape (batch_size, 1) containing the worker-specific weights.

    Returns:
        torch.Tensor: Tensor of shape (batch_size, input_dim)
    """
    return weight[workers] * outputs


def crowd_layer_vb(
    outputs: torch.Tensor, workers: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """
    CrowdLayer Vb transformation. Adds a worker-specific bias to the input.

    Args:
        outputs (torch.Tensor): Tensor of shape (batch_size, input_dim)
        workers (torch.Tensor): Tensor of shape (batch_size,) containing the worker IDs.
        weight (torch.Tensor): Tensor of shape (batch_size, 1) containing the worker-specific biases.

    Returns:
        torch.Tensor: Tensor of shape (batch_size, input_dim)
    """
    return outputs + weight[workers]


def crowd_layer_vw_b(
    outputs: torch.Tensor,
    workers: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    CrowdLayer VW + b transformation. A linear transformation of the input with the bias.

    Args:
        outputs (torch.Tensor): Tensor of shape (batch_size, input_dim)
        workers (torch.Tensor): Tensor of shape (batch_size,) containing the worker IDs.
        scale (torch.Tensor): Tensor of shape (batch_size, 1) containing the worker-specific weights.
        bias (torch.Tensor): Tensor of shape (batch_size, 1) containing the worker-specific biases.

    Returns:
        torch.Tensor: Tensor of shape (batch_size, input_dim)
    """
    return scale[workers] * outputs + bias[workers]


class CrowdLayer(nn.Module):
    """
    CrowdLayer module for classification tasks.

    This method applies a worker-specific transformation of the logits. There are four types of transformations:
    - MW: Multiplication on the worker's confusion matrix.
    - VW: Element-wise multiplication with the worker's weight vector.
    - VB: Element-wise addition with the worker's bias vector.
    - VW + b: Combination of VW and VB: VW * logits + b.

    Filipe Rodrigues and Francisco Pereira. Deep Learning from Crowds.
    *Proceedings of the AAAI Conference on Artificial Intelligence, 32(1),* 2018.
    https://doi.org/10.1609/aaai.v32i1.11506

    Examples:
        >>> from crowdkit.learning import CrowdLayer
        >>> import torch
        >>> input = torch.randn(3, 5)
        >>> workers = torch.tensor([0, 1, 0])
        >>> cl = CrowdLayer(5, 2, conn_type="mw")
        >>> cl(input, workers)
    """

    def __init__(
        self,
        num_labels: int,
        n_workers: int,
        conn_type: str = "mw",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Args:
            num_labels (int): Number of classes.
            n_workers (int): Number of workers.
            conn_type (str): Connection type. One of 'mw', 'vw', 'vb', 'vw+b'.
            device (torch.DeviceObjType): Device to use.
            dtype (torch.dtype): Data type to use.
        Raises:
            ValueError: If conn_type is not one of 'mw', 'vw', 'vb', 'vw+b'.
        """
        super(CrowdLayer, self).__init__()
        self.conn_type = conn_type

        self.n_workers = n_workers
        if conn_type == "mw":
            self.weight = nn.Parameter(
                batch_identity_matrices(
                    n_workers, num_labels, dtype=dtype, device=device
                )
            )
        elif conn_type == "vw":
            self.weight = nn.Parameter(
                torch.ones(n_workers, num_labels, dtype=dtype, device=device)
            )
        elif conn_type == "vb":
            self.weight = nn.Parameter(
                torch.zeros(n_workers, num_labels, dtype=dtype, device=device)
            )
        elif conn_type == "vw+b":
            self.scale = nn.Parameter(
                torch.ones(n_workers, num_labels, dtype=dtype, device=device)
            )
            self.bias = nn.Parameter(
                torch.zeros(n_workers, num_labels, dtype=dtype, device=device)
            )
        else:
            raise ValueError("Unknown connection type for CrowdLayer.")

    def forward(self, outputs: torch.Tensor, workers: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            outputs (torch.Tensor): Tensor of shape (batch_size, input_dim)
            workers (torch.Tensor): Tensor of shape (batch_size,) containing the worker IDs.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, num_labels)
        """
        if self.conn_type == "mw":
            return crowd_layer_mw(outputs, workers, self.weight)
        elif self.conn_type == "vw":
            return crowd_layer_vw(outputs, workers, self.weight)
        elif self.conn_type == "vb":
            return crowd_layer_vb(outputs, workers, self.weight)
        elif self.conn_type == "vw+b":
            return crowd_layer_vw_b(outputs, workers, self.scale, self.bias)
        else:
            raise ValueError("Unknown connection type for CrowdLayer.")
