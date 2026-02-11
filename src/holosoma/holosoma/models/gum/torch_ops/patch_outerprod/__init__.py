from typing import Tuple

import torch

from covariant.learning.validation_utils import check_tensor
from covariant.torch_ops import patch_outerprod_cuda


def patch_outerprod(features: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Compute the centered and normed outer product of a feature tensor on each squared patch.

    For each pixel p_i in the tensor, we take its neighbours N_ij on the kernel_size x kernel_size patch centered in p_ij
    and compute the normalized outer product sum_{q in N_ij} frac{(p_ij -q)^T (p_ij - q)}{||p_i - q||^2}. The previous
    outer product defines the components (b, :, :, i, j).

    Parameters
    ----------
    features: Tensor[DeviceT, FloatX, SHAPE_BCHW]
        Input feature tensor
    kernel_size: int = 3
        Size of the neighborhood on which compute the outer produt

    Returns
    -------
    Tensor[DeviceT, FloatX, Tuple[B, C, C, H, W]]
    """
    check_tensor(features, shape=[None, None, None, None])
    assert kernel_size % 2 == 1, "The kernel size must be an odd number!"
    assert kernel_size > 1, "The kernel should be larger than 1!"

    if torch._C._get_tracing_state():
        (normed_outer_prod,) = torch.ops.custom.patch_outerprod(features, kernel_size)
        return normed_outer_prod
    else:
        return PatchOuterProd.apply(features, kernel_size)


class PatchOuterProd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, kernel_size):
        (normed_outer_prod,) = patch_outerprod_cuda.forward(features, kernel_size)
        ctx.save_for_backward(features)
        ctx.kernel_size = kernel_size
        return normed_outer_prod

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (features,) = ctx.saved_tensors
        (grad_feats,) = patch_outerprod_cuda.backward(grad_output, features, ctx.kernel_size)
        return grad_feats, None
