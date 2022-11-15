import torch
import torch.nn.functional as F


def differentiable_ds(outputs, confusion_matrices):
    normalized_matrices = F.softmax(confusion_matrices, dim=-1).transpose(2, 1)
    return torch.log(
        torch.einsum(
            "lij,ljk->lik",
            normalized_matrices,
            F.softmax(outputs, dim=-1).unsqueeze(-1),
        ).squeeze()
    )


def batch_identity_matrices(batch_size, dim_size, device=None, dtype=None):
    factory_kwargs = {"device": device, "dtype": dtype}
    x = torch.eye(dim_size, **factory_kwargs)
    x = x.reshape((1, dim_size, dim_size))
    return x.repeat(batch_size, 1, 1)
