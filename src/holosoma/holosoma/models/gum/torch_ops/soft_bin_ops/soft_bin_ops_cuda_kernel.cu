#ifdef TORCH_EXTENSION_NAME
#include <torch/extension.h>
#else
#include <torch/script.h>
#endif

#include <vector>

#include "../my_common.h"

namespace {

template <typename scalar_t>
__global__ void kernel_forward_kl_from_gaussian(int world_size,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> loss,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_loss,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> log_probs,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> targets,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> bin_centers, float target_stddev,
    int b_size, int k_size, int h_size, int w_size, int khw_size, int hw_size) {
  int rank = blockIdx.x * blockDim.x + threadIdx.x;

  if (rank < world_size) {
    int b_idx = rank / hw_size;
    int hw_idx = rank % hw_size;
    int h_idx = hw_idx / w_size;
    int w_idx = hw_idx % w_size;

    scalar_t loss_tmp = 0.f;
    scalar_t z_mid, cdf_delta;
    scalar_t bin_width =
        (bin_centers[b_idx][k_size - 1][h_idx][w_idx] - bin_centers[b_idx][0][h_idx][w_idx]) / (k_size - 1);
    scalar_t my_target = targets[b_idx][h_idx][w_idx];

    if (my_target < bin_centers[b_idx][1][h_idx][w_idx]) {
      cdf_delta = (bin_centers[b_idx][1][h_idx][w_idx] - my_target) / bin_width;
      if (cdf_delta > 1e-12) {
        grad_loss[b_idx][0][h_idx][w_idx] = -cdf_delta;
        loss_tmp += cdf_delta * (log(cdf_delta) - log_probs[b_idx][0][h_idx][w_idx]);
      }

      cdf_delta = (my_target - bin_centers[b_idx][0][h_idx][w_idx]) / bin_width;
      if (cdf_delta > 1e-12) {
        grad_loss[b_idx][1][h_idx][w_idx] = -cdf_delta;
        loss_tmp += cdf_delta * (log(cdf_delta) - log_probs[b_idx][1][h_idx][w_idx]);
      }

    } else if (my_target > bin_centers[b_idx][k_size - 2][h_idx][w_idx]) {
      cdf_delta = (bin_centers[b_idx][k_size - 1][h_idx][w_idx] - my_target) / bin_width;
      if (cdf_delta > 1e-12) {
        grad_loss[b_idx][k_size - 2][h_idx][w_idx] = -cdf_delta;
        loss_tmp += cdf_delta * (log(cdf_delta) - log_probs[b_idx][k_size - 2][h_idx][w_idx]);
      }

      cdf_delta = (my_target - bin_centers[b_idx][k_size - 2][h_idx][w_idx]) / bin_width;
      if (cdf_delta > 1e-12) {
        grad_loss[b_idx][k_size - 1][h_idx][w_idx] = -cdf_delta;
        loss_tmp += cdf_delta * (log(cdf_delta) - log_probs[b_idx][k_size - 1][h_idx][w_idx]);
      }

    } else {
      for (int k_idx = 0; k_idx < k_size; k_idx++) {
        z_mid = (bin_centers[b_idx][k_idx][h_idx][w_idx] - my_target) / (target_stddev * bin_width);
        cdf_delta = normcdf(z_mid + (0.5 / target_stddev)) - normcdf(z_mid - (0.5 / target_stddev));
        if (cdf_delta > 1e-12) {
          loss_tmp += cdf_delta * (log(cdf_delta) - log_probs[b_idx][k_idx][h_idx][w_idx]);
          grad_loss[b_idx][k_idx][h_idx][w_idx] = -cdf_delta;
        }
      }
    }
    loss[b_idx][h_idx][w_idx] = loss_tmp;
  }
}

template <typename scalar_t>
__global__ void kernel_forward_peaky_attention(int world_size,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> output,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> logits,
    const torch::PackedTensorAccessor32<long, 3, torch::RestrictPtrTraits> max_idx,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> bin_centers, int bin_ksize, int b_size,
    int k_size, int h_size, int w_size, int khw_size, int hw_size) {
  int rank = blockIdx.x * blockDim.x + threadIdx.x;

  if (rank < world_size) {
    int b_idx = rank / hw_size;
    int hw_idx = rank % hw_size;
    int h_idx = hw_idx / w_size;
    int w_idx = hw_idx % w_size;

    scalar_t value = 0.f, denom = 0.f, exp_logit, max_logit = -std::numeric_limits<float>::infinity();
    int k_idx = max_idx[b_idx][h_idx][w_idx];
    max_logit = logits[b_idx][k_idx][h_idx][w_idx];

    // do softmax(logits - max_logit) to avoid overflow
    for (int k_idx = max_idx[b_idx][h_idx][w_idx] - bin_ksize; k_idx <= max_idx[b_idx][h_idx][w_idx] + bin_ksize;
         k_idx++) {
      if (0 <= k_idx && k_idx < k_size) {
        exp_logit = expf(logits[b_idx][k_idx][h_idx][w_idx] - max_logit);
        denom += exp_logit;
        value += exp_logit * bin_centers[b_idx][k_idx][h_idx][w_idx];
      }
    }
    output[b_idx][h_idx][w_idx] = value / denom;
  }
}
}  // namespace

std::vector<at::Tensor> kl_from_gaussian_cuda_forward(
    at::Tensor log_probs, at::Tensor targets, at::Tensor bin_centers, float target_stddev) {
  auto b_size = log_probs.size(0);
  auto k_size = log_probs.size(1);
  auto h_size = log_probs.size(2);
  auto w_size = log_probs.size(3);

  auto loss = at::zeros({b_size, h_size, w_size}, log_probs.options());
  auto grad_loss = at::zeros_like(log_probs);

  auto world_size = b_size * h_size * w_size;
  const int threads = 1024;

  if (bin_centers.dim() == 2) {
    // Makes `bin_centers` have the same shape as `logits`, but don't copy the data.
    bin_centers = bin_centers.unsqueeze(-1).unsqueeze(-1).expand_as(log_probs);
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(log_probs.scalar_type(), "kl_from_gaussian_forward", ([&] {
    kernel_forward_kl_from_gaussian<<<(world_size + threads - 1) / threads, threads>>>(world_size,
        loss.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        grad_loss.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
        log_probs.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
        targets.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        bin_centers.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), target_stddev, b_size, k_size, h_size,
        w_size, k_size * h_size * w_size, h_size * w_size);
  }));

  return {loss, grad_loss};
}

std::vector<at::Tensor> peaky_attention_cuda_forward(
    at::Tensor logits, at::Tensor max_idx, at::Tensor bin_centers, int bin_ksize) {
  auto b_size = logits.size(0);
  auto k_size = logits.size(1);
  auto h_size = logits.size(2);
  auto w_size = logits.size(3);

  auto output = at::zeros({b_size, h_size, w_size}, logits.options());

  auto world_size = b_size * h_size * w_size;
  const int threads = 1024;

  if (bin_centers.dim() == 2) {
    // Makes `bin_centers` have the same shape as `logits`, but don't copy the data.
    bin_centers = bin_centers.unsqueeze(-1).unsqueeze(-1).expand_as(logits);
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(logits.scalar_type(), "peaky_attention", ([&] {
    kernel_forward_peaky_attention<<<(world_size + threads - 1) / threads, threads>>>(world_size,
        output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        logits.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
        max_idx.packed_accessor32<long, 3, torch::RestrictPtrTraits>(),
        bin_centers.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), bin_ksize, b_size, k_size, h_size,
        w_size, k_size * h_size * w_size, h_size * w_size);
  }));

  return {output};
}
