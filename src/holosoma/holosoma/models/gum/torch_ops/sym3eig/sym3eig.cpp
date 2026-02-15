#ifdef TORCH_EXTENSION_NAME
#include <torch/extension.h>
#else
#include <torch/script.h>
#endif

std::tuple<at::Tensor, at::Tensor> sym3eig_cuda_forward(at::Tensor x);
std::tuple<at::Tensor, at::Tensor> sym3eig_cpu_forward(at::Tensor x);

at::Tensor sym3eig_cuda_backward(
    at::Tensor eig_vec_grad, at::Tensor eig_vec, at::Tensor eig_val_grad, at::Tensor eig_val);
at::Tensor sym3eig_cpu_backward(
    at::Tensor eig_vec_grad, at::Tensor eig_vec, at::Tensor eig_val_grad, at::Tensor eig_val);

#include "../my_common.h"

std::tuple<at::Tensor, at::Tensor> sym3eig_forward(at::Tensor x) {
  CHECK_CONTIGUOUS(x);
  DISPATCH_CPU_OR_CUDA(x.device(), sym3eig_cpu_forward, sym3eig_cuda_forward, x);
}

at::Tensor sym3eig_backward(at::Tensor eig_vec_grad, at::Tensor eig_vec, at::Tensor eig_val_grad, at::Tensor eig_val) {
  CHECK_CONTIGUOUS(eig_vec_grad);
  CHECK_CONTIGUOUS(eig_vec);
  CHECK_CONTIGUOUS(eig_val_grad);
  CHECK_CONTIGUOUS(eig_val);
  CHECK_SAME_DEVICE(eig_vec_grad, eig_vec);
  CHECK_SAME_DEVICE(eig_vec_grad, eig_val_grad);
  CHECK_SAME_DEVICE(eig_vec_grad, eig_val);
  DISPATCH_CPU_OR_CUDA(
      eig_vec_grad.device(), sym3eig_cpu_backward, sym3eig_cuda_backward, eig_vec_grad, eig_vec, eig_val_grad, eig_val);
}

#ifdef TORCH_EXTENSION_NAME
static auto registry = torch::RegisterOperators("custom::sym3eig", &sym3eig_forward);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sym3eig_forward, "sym3eig forward()");
  m.def("backward", &sym3eig_backward, "sym3eig backward()");
}
#else
TORCH_LIBRARY_FRAGMENT(custom, m) {
  m.def("sym3eig", sym3eig_forward);
}
#endif
