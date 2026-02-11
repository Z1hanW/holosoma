from typing import Tuple, Union

import torch

from covariant.learning.validation_utils import check_tensor
from covariant.torch_ops import soft_bin_ops_cuda


class KlGaussianTarget(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_probs, targets, bin_centers, target_stddev):
        loss, grad_loss = torch.ops.custom.kl_from_gaussian_forward(log_probs, targets, bin_centers, target_stddev)
        ctx.save_for_backward(grad_loss)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        (grad_loss,) = ctx.saved_tensors
        return grad_output.unsqueeze(1) * grad_loss, None, None, None


def kl_from_gaussian(logprobs, targets, bin_centers, target_stddev):
    """Compute the same value as `kl_from_gaussian_with_logits()`, except that `logits` have already been normalized to `logprobs`."""
    b, d, h, w = logprobs.shape
    check_tensor(logprobs, dtype=torch.float32)
    check_tensor(targets, dtype=torch.float32, shape=[b, h, w])

    if bin_centers.dim() == 2:
        check_tensor(bin_centers, dtype=torch.float32, shape=[b, d])
    elif bin_centers.dim() == 4:
        check_tensor(bin_centers, dtype=torch.float32, shape=[b, d, h, w])
    else:
        raise ValueError(
            f"Expected `bin_centers` to have either 2 or 4 dims, but got Tensor of shape: {bin_centers.shape}"
        )

    assert torch.allclose(torch.sort(bin_centers, dim=1)[0], bin_centers)
    return KlGaussianTarget.apply(logprobs, targets, bin_centers, target_stddev)


def peaky_attention(logits: torch.Tensor, bin_centers: torch.Tensor, bin_ksize: int = 2) -> torch.Tensor:
    """Find the most likely mode of a soft-bin distribution.

    This is done by taking a weighted sum of the bins according to their softmax probabilities,
    but only considering the ones within `bin_ksize` of the maximum logit (on either side).

    Parameters
    ----------
    logits : Tensor[DeviceT, Float32, Tuple[B, D, H, W]]
    bin_centers : Tensor[DeviceT, Float32, Union[Tuple[B, D], Tuple[B, D, H, W]]]
    bin_ksize : int, optional

    Returns
    -------
    Tensor[DeviceT, FloatX, Tuple[B, H, W]]
    """
    check_tensor(logits, shape=[None, None, None, None])
    b, d, h, w = logits.shape

    if bin_centers.dim() == 2:
        check_tensor(bin_centers, dtype=logits.dtype, shape=[b, d])
    elif bin_centers.dim() == 4:
        check_tensor(bin_centers, dtype=logits.dtype, shape=[b, d, h, w])
    else:
        raise ValueError(
            f"Expected `bin_centers` to have either 2 or 4 dims, but got Tensor of shape: {bin_centers.shape}"
        )

    (peaky_pred,) = torch.ops.custom.peaky_attention_forward(logits, logits.argmax(1), bin_centers, bin_ksize)
    return peaky_pred


def kl_from_gaussian_with_logits(
    logits: torch.Tensor, targets: torch.Tensor, bin_centers: torch.Tensor, target_stddev: float = 0.5
) -> torch.Tensor:
    """Compute the KL divergence between a discretized (diagonal) gaussian and a categorical distribution.

    To save GPU memory, the target probabilities are never explicitly constructed.
    To avoid a loss of numerical precision, this op should not be called in float16.

    Parameters
    ----------
    logits : Tensor[DeviceT, Float32, Tuple[B, D, H, W]]
    targets : Tensor[DeviceT, Float32, Tuple[B, H, W]]
        The mean of the gaussian distribution.
    bin_centers : Tensor[DeviceT, Float32, Union[Tuple[B, D], Tuple[B, D, H, W]]]
        These should be equally spaced and increasing along the `D` axis.
    target_stddev : float, optional.
        The standard deviation of the gaussian distriibution, expressed as a fraction of the bin width (defaults to 0.5).
        This means that ~95% of the probability mass will fall over `2.0 / target_stddev` bins.
        Currently, the same standard deviation must be used for each element of `bin_ceneters`.

    Returns
    -------
    Tensor[DeviceT, Float32, Tuple[B, H, W]]
    """
    logprobs = torch.log_softmax(logits, dim=1)
    return kl_from_gaussian(logprobs, targets, bin_centers, target_stddev=target_stddev)
