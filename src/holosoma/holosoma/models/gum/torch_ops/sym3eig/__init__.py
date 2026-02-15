from typing import Tuple

import torch

from covariant.learning.torch_misc_utils import amp_autocast_if_cuda
from covariant.learning.validation_utils import check_tensor
from covariant.torch_ops import sym3eig_cuda


@amp_autocast_if_cuda(enabled=False)
def sym3eig(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the eigen values and vectors of a batch of symmetric 3x3 matrices.

    If the matrices are not symmetric, it transforms it into their symmetric version, i.e., (x + x.T)/2.

    Parameters
    ----------
    x: Tensor[DeviceT, FloatX, SHAPE_B33]

    Returns
    -------
    Tuple[Tensor[DeviceT, FloatX, SHAPE_B3], Tensor[DeviceT, FloatX, SHAPE_B33]]
    Eigen values, eigen vectors.
    """
    check_tensor(x, shape=[None, 3, 3])
    if torch._C._get_tracing_state():
        x = x.contiguous()
        return torch.ops.custom.sym3eig(x)
    else:
        return Sym3Eig.apply(x)


class Sym3Eig(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = (x + torch.transpose(x, 1, 2)) / 2
        x = x.contiguous()
        eig_val, eig_vec = sym3eig_cuda.forward(x)
        ctx.save_for_backward(eig_val, eig_vec)
        return eig_val, eig_vec

    @staticmethod
    def backward(ctx, eig_val_grad, eig_vec_grad):
        eig_val, eig_vec = ctx.saved_tensors
        eig_val_grad = eig_val_grad.contiguous()
        eig_vec_grad = eig_vec_grad.contiguous()
        grad_x = sym3eig_cuda.backward(eig_vec_grad, eig_vec, eig_val_grad, eig_val)
        grad_x = (grad_x + torch.transpose(grad_x, 1, 2)) / 2
        """
        # Version using pytorch ops
        ut = torch.transpose(eig_vec, 1, 2)
        u = eig_vec
        gu = eig_vec_grad
        s = eig_val
        gs = eig_val_grad
        gs = torch.Tensor([gs[:,0],.0,.0,.0,gs[:,1],.0,.0,.0,gs[:,2]]).view(-1,3,3).double()
        F = 1/(s.unsqueeze(1).expand(-1,3,-1) - s.unsqueeze(2).expand(-1,-1,3))
        F[:,0,0] = F[:,1,1] = F[:,2,2] = 0.0
        X = torch.matmul(ut, gu)
        X = F*X
        grad_matrices = torch.matmul(u, torch.matmul(X, ut))
        val = torch.matmul(u, torch.matmul(gs, ut))
        grad_matrices = grad_matrices + val
        """
        grad_x[torch.isnan(grad_x)] = 0.0
        return grad_x
