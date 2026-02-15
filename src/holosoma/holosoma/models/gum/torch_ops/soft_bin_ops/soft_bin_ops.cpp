#ifdef TORCH_EXTENSION_NAME
#include <torch/extension.h>
#else
#include <torch/script.h>
#endif

#include <vector>

std::vector<at::Tensor> kl_from_gaussian_cuda_forward(
    at::Tensor log_probs, at::Tensor targets, at::Tensor bin_centers, float target_stddev);

std::vector<at::Tensor> kl_from_gaussian_cpu_forward(
    at::Tensor log_probs, at::Tensor targets, at::Tensor bin_centers, float target_stddev);

std::vector<at::Tensor> peaky_attention_cuda_forward(
    at::Tensor logits, at::Tensor max_idx, at::Tensor bin_centers, int bin_ksize);

std::vector<at::Tensor> peaky_attention_cpu_forward(
    at::Tensor logits, at::Tensor max_idx, at::Tensor bin_centers, int bin_ksize);

#include "../my_common.h"

std::vector<at::Tensor> kl_from_gaussian_forward(
    at::Tensor log_probs, at::Tensor targets, at::Tensor bin_centers, double target_stddev) {
  CHECK_SAME_DEVICE(log_probs, targets);
  CHECK_SAME_DEVICE(log_probs, bin_centers);
  DISPATCH_CPU_OR_CUDA(log_probs.device(), kl_from_gaussian_cpu_forward, kl_from_gaussian_cuda_forward, log_probs,
      targets, bin_centers, target_stddev);
}

std::vector<at::Tensor> peaky_attention_forward(
    at::Tensor logits, at::Tensor max_idx, at::Tensor bin_centers, int64_t bin_ksize) {
  CHECK_SAME_DEVICE(logits, bin_centers);
  CHECK_SAME_DEVICE(logits, max_idx);
  DISPATCH_CPU_OR_CUDA(logits.device(), peaky_attention_cpu_forward, peaky_attention_cuda_forward, logits, max_idx,
      bin_centers, bin_ksize);
}

#ifdef TORCH_EXTENSION_NAME
static auto registry =
    torch::RegisterOperators()
        .op("custom::kl_from_gaussian_forward(Tensor log_probs, Tensor targets, Tensor bin_centers, float target_stddev) -> Tensor[]",
            &kl_from_gaussian_forward)
        .op("custom::peaky_attention_forward(Tensor logits, Tensor max_idx, Tensor bin_centers, int bin_ksize) -> Tensor[]",
            &peaky_attention_forward);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kl_from_gaussian_forward", &kl_from_gaussian_forward, "kl_from_gaussian forward()");
  m.def("peaky_attention_forward", &peaky_attention_forward, "peaky_attention forward()");
}
#else
TORCH_LIBRARY_FRAGMENT(custom, m) {
  m.def("kl_from_gaussian_forward", kl_from_gaussian_forward);
  m.def("peaky_attention_forward", peaky_attention_forward);
}
#endif
