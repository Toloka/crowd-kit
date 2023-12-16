import pytest
import torch

from crowdkit.learning import CoNAL

from .data_learning import toy_logits, toy_workers  # noqa F401


@pytest.mark.filterwarnings("ignore:Lazy modules are a new feature")
def test_conal(
    toy_logits: torch.Tensor, toy_workers: torch.Tensor  # noqa F811
) -> None:
    embeddings = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
        ]
    )
    conal = CoNAL(5, 2, com_emb_size=20, user_feature=None)
    out = conal(embeddings, toy_logits, toy_workers)
    assert out.shape == (3, 5)
    out.sum().backward()
    assert conal.annotator_confusion_matrices.grad.shape == (2, 5, 5)
    assert conal.common_confusion_matrix.grad.shape == (5, 5)
    # I'm not sure that we can properly test the values of the output because the module is randomly initialized.
