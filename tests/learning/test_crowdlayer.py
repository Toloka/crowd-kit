import torch

from crowdkit.learning import CrowdLayer

from .data_learning import toy_logits, toy_workers  # noqa F401


def test_crowdlayer_vb(
    toy_logits: torch.Tensor, toy_workers: torch.Tensor  # noqa F811
) -> None:
    crowd_layer = CrowdLayer(5, 2, conn_type="vb")
    out = crowd_layer(toy_logits, toy_workers)
    assert out.shape == (3, 5)
    assert torch.allclose(out, toy_logits)
    out.sum().backward()
    assert crowd_layer.weight.grad.shape == (2, 5)  # type: ignore


def test_crowdlayer_vw(
    toy_logits: torch.Tensor, toy_workers: torch.Tensor  # noqa F811
) -> None:
    crowd_layer = CrowdLayer(5, 2, conn_type="vw")
    out = crowd_layer(toy_logits, toy_workers)
    assert out.shape == (3, 5)
    assert torch.allclose(out, toy_logits)
    out.sum().backward()
    assert crowd_layer.weight.grad.shape == (2, 5)  # type: ignore


def test_crowdlayer_vw_b(
    toy_logits: torch.Tensor, toy_workers: torch.Tensor  # noqa F811
) -> None:
    crowd_layer = CrowdLayer(5, 2, conn_type="vw+b")
    out = crowd_layer(toy_logits, toy_workers)
    assert out.shape == (3, 5)
    assert torch.allclose(out, toy_logits)
    out.sum().backward()
    assert crowd_layer.scale.grad.shape == (2, 5)  # type: ignore
    assert crowd_layer.bias.grad.shape == (2, 5)  # type: ignore


def test_crowdlayer_mw(
    toy_logits: torch.Tensor, toy_workers: torch.Tensor  # noqa F811
) -> None:
    crowd_layer = CrowdLayer(5, 2, conn_type="mw")
    out = crowd_layer(toy_logits, toy_workers)
    assert out.shape == (3, 5)
    assert torch.allclose(out, toy_logits)
    out.sum().backward()
    assert crowd_layer.weight.grad.shape == (2, 5, 5)  # type: ignore
