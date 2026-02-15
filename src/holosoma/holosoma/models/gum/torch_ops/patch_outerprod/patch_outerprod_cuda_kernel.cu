#ifdef TORCH_EXTENSION_NAME
#include <torch/extension.h>
#else
#include <torch/script.h>
#endif

#include <THC/THCAtomics.cuh>
#include <vector>

#include "../my_common.h"

namespace {

template <typename scalar_t>
__global__ void patch_outerprod_forward_kernel(int world_size,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> output,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> feat_tensor, int kernel_size, int b_size,
    int c_size, int h_size, int w_size, int kernel_squared, int hw_size) {
  int rank = blockIdx.x * blockDim.x + threadIdx.x;

  if (rank < world_size) {
    int b_idx = rank / hw_size;
    int hw_idx = rank % hw_size;
    int h_idx = hw_idx / w_size;
    int w_idx = hw_idx % w_size;

    for (int k_idx = 0; k_idx < kernel_squared; ++k_idx) {
      int delta_h = k_idx / kernel_size - (kernel_size - 1) / 2;
      int delta_w = k_idx % kernel_size - (kernel_size - 1) / 2;
      int other_h_idx = h_idx + delta_h;
      int other_w_idx = w_idx + delta_w;

      if (0 <= other_h_idx && other_h_idx < h_size && 0 <= other_w_idx && other_w_idx < w_size &&
          (delta_h != 0 || delta_w != 0)) {
        scalar_t norm_squared = 0.0;
        // compute the norm of the vec
        for (int c_idx = 0; c_idx < c_size; c_idx++) {
          scalar_t vec = feat_tensor[b_idx][c_idx][h_idx][w_idx] - feat_tensor[b_idx][c_idx][other_h_idx][other_w_idx];
          norm_squared += powf(vec, 2);
        }

        // compute the outer prod
        for (int ci_idx = 0; ci_idx < c_size; ci_idx++) {
          for (int cj_idx = 0; cj_idx < c_size; cj_idx++) {
            scalar_t vec_i =
                feat_tensor[b_idx][ci_idx][h_idx][w_idx] - feat_tensor[b_idx][ci_idx][other_h_idx][other_w_idx];
            scalar_t vec_j =
                feat_tensor[b_idx][cj_idx][h_idx][w_idx] - feat_tensor[b_idx][cj_idx][other_h_idx][other_w_idx];
            output[b_idx][ci_idx][cj_idx][h_idx][w_idx] += (vec_i * vec_j) / (norm_squared + 1e-8);
          }
        }
      }
    }
  }
}

template <typename scalar_t>
__global__ void patch_outerprod_backward_kernel(int world_size,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> grad_output,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> feat_tensor,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_feats, int kernel_size, int b_size,
    int c_size, int h_size, int w_size, int kernel_squared, int hw_size) {
  int rank = blockIdx.x * blockDim.x + threadIdx.x;
  if (rank < world_size) {
    int b_idx = rank / hw_size;
    int hw_idx = rank % hw_size;
    int h_idx = hw_idx / w_size;
    int w_idx = hw_idx % w_size;

    for (int k_idx = 0; k_idx < kernel_squared; ++k_idx) {
      int delta_h = k_idx / kernel_size - (kernel_size - 1) / 2;
      int delta_w = k_idx % kernel_size - (kernel_size - 1) / 2;
      int other_h_idx = h_idx + delta_h;
      int other_w_idx = w_idx + delta_w;

      if (0 <= other_h_idx && other_h_idx < h_size && 0 <= other_w_idx && other_w_idx < w_size &&
          (delta_h != 0 || delta_w != 0)) {
        scalar_t norm_squared = 0.0;
        // compute the norm of the vec
        for (int c_idx = 0; c_idx < c_size; c_idx++) {
          scalar_t vec = feat_tensor[b_idx][c_idx][h_idx][w_idx] - feat_tensor[b_idx][c_idx][other_h_idx][other_w_idx];
          norm_squared += powf(vec, 2);
        }
        norm_squared += 1e-8;

        // compute the outer prod
        for (int c_idx = 0; c_idx < c_size; c_idx++) {
          for (int ci_idx = 0; ci_idx < c_size; ci_idx++) {
            for (int cj_idx = 0; cj_idx < c_size; cj_idx++) {
              scalar_t grad_feat;
              scalar_t vec_i =
                  feat_tensor[b_idx][ci_idx][h_idx][w_idx] - feat_tensor[b_idx][ci_idx][other_h_idx][other_w_idx];
              scalar_t vec_j =
                  feat_tensor[b_idx][cj_idx][h_idx][w_idx] - feat_tensor[b_idx][cj_idx][other_h_idx][other_w_idx];
              if (c_idx == ci_idx && c_idx == cj_idx) {
                grad_feat = 2 * vec_i / norm_squared * (1 - powf(vec_i, 2) / norm_squared);
              }
              if (c_idx == ci_idx && c_idx != cj_idx) {
                grad_feat = vec_j / norm_squared * (1 - 2 * powf(vec_i, 2) / norm_squared);
              }
              if (c_idx == cj_idx && c_idx != ci_idx) {
                grad_feat = vec_i / norm_squared * (1 - 2 * powf(vec_j, 2) / norm_squared);
              }
              if (c_idx != ci_idx && c_idx != cj_idx) {
                scalar_t vec_ =
                    feat_tensor[b_idx][c_idx][h_idx][w_idx] - feat_tensor[b_idx][c_idx][other_h_idx][other_w_idx];
                grad_feat = -2 * vec_ * vec_i * vec_j / powf(norm_squared, 2);
              }
              scalar_t grad_out = grad_output[b_idx][ci_idx][cj_idx][h_idx][w_idx];
              atomicAdd(&grad_feats[b_idx][c_idx][h_idx][w_idx], grad_feat * grad_out);
              atomicAdd(&grad_feats[b_idx][c_idx][other_h_idx][other_w_idx], -grad_feat * grad_out);
            }
          }
        }
      }
    }
  }
}
}  // namespace

std::vector<at::Tensor> patch_outerprod_cuda_forward(at::Tensor feat_tensor, int kernel_size) {
  auto b_size = feat_tensor.size(0);
  auto c_size = feat_tensor.size(1);
  auto h_size = feat_tensor.size(2);
  auto w_size = feat_tensor.size(3);
  const int threads = 1024;

  auto output = at::zeros({b_size, c_size, c_size, h_size, w_size}, feat_tensor.options());

  auto world_size = b_size * h_size * w_size;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(feat_tensor.scalar_type(), "patch_outerprod", ([&] {
    patch_outerprod_forward_kernel<<<(world_size + threads - 1) / threads, threads>>>(world_size,
        output.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
        feat_tensor.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), kernel_size, b_size, c_size, h_size,
        w_size, kernel_size * kernel_size, h_size * w_size);
  }));
  return {output};
}

std::vector<at::Tensor> patch_outerprod_cuda_backward(at::Tensor grad_output, at::Tensor feat_tensor, int kernel_size) {
  auto b_size = feat_tensor.size(0);
  auto c_size = feat_tensor.size(1);
  auto h_size = feat_tensor.size(2);
  auto w_size = feat_tensor.size(3);
  const int threads = 1024;

  auto world_size = b_size * h_size * w_size;

  auto grad_feats = at::zeros({b_size, c_size, h_size, w_size}, feat_tensor.options());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(feat_tensor.scalar_type(), "patch_outerprod", ([&] {
    patch_outerprod_backward_kernel<<<(world_size + threads - 1) / threads, threads>>>(world_size,
        grad_output.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
        feat_tensor.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
        grad_feats.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), kernel_size, b_size, c_size, h_size,
        w_size, kernel_size * kernel_size, h_size * w_size);
  }));

  return {grad_feats};
}
