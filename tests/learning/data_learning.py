import pytest
import torch


@pytest.fixture
def toy_logits() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 2.0],
        ]
    )


@pytest.fixture
def toy_workers() -> torch.Tensor:
    return torch.tensor([0, 1, 0])
