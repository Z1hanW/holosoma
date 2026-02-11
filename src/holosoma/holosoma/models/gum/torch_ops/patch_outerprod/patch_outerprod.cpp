#ifdef TORCH_EXTENSION_NAME
#include <torch/extension.h>
#else
#include <torch/script.h>
#endif

#include <vector>

std::vector<at::Tensor> patch_outerprod_cuda_forward(at::Tensor feat_tensor, int kernel_size);
std::vector<at::Tensor> patch_outerprod_cpu_forward(at::Tensor feat_tensor, int kernel_size);

std::vector<at::Tensor> patch_outerprod_cuda_backward(at::Tensor grad_output, at::Tensor feat_tensor, int kernel_size);
std::vector<at::Tensor> patch_outerprod_cpu_backward(at::Tensor grad_output, at::Tensor feat_tensor, int kernel_size);

#include "../my_common.h"

std::vector<at::Tensor> patch_outerprod_forward(at::Tensor feat_tensor, int64_t kernel_size) {
  DISPATCH_CPU_OR_CUDA(
      feat_tensor.device(), patch_outerprod_cpu_forward, patch_outerprod_cuda_forward, feat_tensor, kernel_size);
}

std::vector<at::Tensor> patch_outerprod_backward(at::Tensor grad_output, at::Tensor feat_tensor, int64_t kernel_size) {
  CHECK_SAME_DEVICE(grad_output, feat_tensor);
  DISPATCH_CPU_OR_CUDA(feat_tensor.device(), patch_outerprod_cpu_backward, patch_outerprod_cuda_backward, grad_output,
      feat_tensor, kernel_size);
}

#ifdef TORCH_EXTENSION_NAME
static auto registry = torch::RegisterOperators("custom::patch_outerprod", &patch_outerprod_forward);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &patch_outerprod_forward, "patch outerprod forward (CUDA and CPU)");
  m.def("backward", &patch_outerprod_backward, "patch outerprod backward (CUDA and CPU)");
}
#else
TORCH_LIBRARY_FRAGMENT(custom, m) {
  m.def("patch_outerprod", patch_outerprod_forward);
}
#endif
