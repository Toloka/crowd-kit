# Adapted from:
# https://github.com/zdchu/CoNAL/blob/main/conal.py
__all__ = [
    "CoNAL",
]

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import nn

from crowdkit.learning.utils import differentiable_ds


def _identity_init(shape: Union[Tuple[int, int], Tuple[int, int, int]]) -> torch.Tensor:
    """
    Creates a tensor containing identity matrices.

    Args:
        shape (Tuple[int]): Tuple of ints representing the shape of the tensor.

    Returns:
        torch.Tensor: Tensor containing identity matrices.
    """
    out = np.zeros(shape, dtype=np.float32)
    if len(shape) == 3:
        for r in range(shape[0]):
            for i in range(shape[1]):
                out[r, i, i] = 2.0
    elif len(shape) == 2:
        for i in range(shape[1]):
            out[i, i] = 2.0
    return torch.Tensor(out)


class CoNAL(nn.Module):
    """
    Common Noise Adaptation Layers (CoNAL). This method introduces two types of confusions: worker-specific and
    global. Each is parameterized by a confusion matrix. The ratio of the two confusions is determined by the
    common noise adaptation layer. The common noise adaptation layer is a trainable function that takes the
    instance embedding and the worker ID as input and outputs a scalar value between 0 and 1.

    Zhendong Chu, Jing Ma, and Hongning Wang. Learning from Crowds by Modeling Common Confusions.
    *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(7), 5832-5840, 2021.
    https://doi.org/10.1609/aaai.v35i7.16730

    Examples:
        >>> from crowdkit.learning import CoNAL
        >>> import torch
        >>> input = torch.randn(3, 5)
        >>> workers = torch.tensor([0, 1, 0])
        >>> embeddings = torch.randn(3, 5)
        >>> conal = CoNAL(5, 2)
        >>> conal(embeddings, input, workers)
    """

    def __init__(
        self,
        num_labels: int,
        n_workers: int,
        com_emb_size: int = 20,
        user_feature: Optional[NDArray[np.float32]] = None,
    ):
        """
        Initializes the CoNAL module.

        Args:
            num_labels (int): Number of classes.
            n_workers (int): Number of annotators.
            com_emb_size (int): Embedding size of the common noise module.
            user_feature (np.ndarray): User feature vector.
        """
        super().__init__()
        self.n_workers = n_workers
        self.annotator_confusion_matrices = nn.Parameter(
            _identity_init((n_workers, num_labels, num_labels)),
            requires_grad=True,
        )

        self.common_confusion_matrix = nn.Parameter(
            _identity_init((num_labels, num_labels)), requires_grad=True
        )

        user_feature = user_feature or np.eye(n_workers, dtype=np.float32)
        self.user_feature_vec = nn.Parameter(
            torch.from_numpy(user_feature).float(), requires_grad=False
        )
        self.diff_linear_1 = nn.LazyLinear(128)
        self.diff_linear_2 = nn.Linear(128, com_emb_size)
        self.user_feature_1 = nn.Linear(self.user_feature_vec.size(1), com_emb_size)

    def simple_common_module(
        self, input: torch.Tensor, workers: torch.Tensor
    ) -> torch.Tensor:
        """
        Common noise adoptation module.

        Args:
            input (torch.Tensor): Tensor of shape (batch_size, embedding_size)
            workers (torch.Tensor): Tensor of shape (batch_size,) containing the worker IDs.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, 1) containing the common noise rate.
        """
        instance_difficulty = self.diff_linear_1(input)
        instance_difficulty = self.diff_linear_2(instance_difficulty)

        instance_difficulty = F.normalize(instance_difficulty)
        user_feature = self.user_feature_1(self.user_feature_vec[workers])
        user_feature = F.normalize(user_feature)
        common_rate = torch.sum(instance_difficulty * user_feature, dim=1)
        common_rate = torch.sigmoid(common_rate).unsqueeze(1)
        return common_rate

    def forward(
        self, embeddings: torch.Tensor, logits: torch.Tensor, workers: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the CoNAL module.

        Args:
            embeddings (torch.Tensor): Tensor of shape (batch_size, embedding_size)
            logits (torch.Tensor): Tensor of shape (batch_size, num_classes)
            workers (torch.Tensor): Tensor of shape (batch_size,) containing the worker IDs.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, 1) containing the predicted output probabilities.
        """
        x = embeddings.view(embeddings.size(0), -1)
        common_rate = self.simple_common_module(x, workers)
        common_prob = torch.einsum(
            "ij,jk->ik", (F.softmax(logits, dim=-1), self.common_confusion_matrix)
        )
        batch_confusion_matrices = self.annotator_confusion_matrices[workers]
        indivi_prob = differentiable_ds(logits, batch_confusion_matrices)
        crowd_out: torch.Tensor = (
            common_rate * common_prob + (1 - common_rate) * indivi_prob
        )  # single instance
        return crowd_out
