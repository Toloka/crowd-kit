import pytest
import torch

@pytest.fixture
def toy_logits() -> torch.Tensor:
    return torch.tensor(
        [
            [1., 2., 3., 4., 5.],
            [1., 1., 1., 1., 1.],
            [2., 2., 2., 2., 2.],
        ]
    )


@pytest.fixture
def toy_workers() -> torch.Tensor:
    return torch.tensor([0, 1, 0])
