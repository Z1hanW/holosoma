#ifdef TORCH_EXTENSION_NAME
#include <torch/extension.h>
#else
#include <torch/script.h>
#endif
// the extension header must be imported first
#include <ATen/Parallel.h>
#include <omp.h>

#include <vector>

template <typename scalar_t>
void patch_outerprod_forward_kernel(int world_size, int rank, torch::TensorAccessor<scalar_t, 5> output,
    const torch::TensorAccessor<scalar_t, 4> feat_tensor, int kernel_size, int b_size, int c_size, int h_size,
    int w_size, int kernel_squared, int hw_size) {
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
          norm_squared += std::pow(vec, 2);
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
void patch_outerprod_backward_kernel(int world_size, int rank, const torch::TensorAccessor<scalar_t, 5> grad_output,
    const torch::TensorAccessor<scalar_t, 4> feat_tensor, torch::TensorAccessor<scalar_t, 4> grad_feats,
    int kernel_size, int b_size, int c_size, int h_size, int w_size, int kernel_squared, int hw_size,
    omp_lock_t* lock) {
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
          norm_squared += std::pow(vec, 2);
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
                grad_feat = 2 * vec_i / norm_squared * (1 - std::pow(vec_i, 2) / norm_squared);
              }
              if (c_idx == ci_idx && c_idx != cj_idx) {
                grad_feat = vec_j / norm_squared * (1 - 2 * std::pow(vec_i, 2) / norm_squared);
              }
              if (c_idx == cj_idx && c_idx != ci_idx) {
                grad_feat = vec_i / norm_squared * (1 - 2 * std::pow(vec_j, 2) / norm_squared);
              }
              if (c_idx != ci_idx && c_idx != cj_idx) {
                scalar_t vec_ =
                    feat_tensor[b_idx][c_idx][h_idx][w_idx] - feat_tensor[b_idx][c_idx][other_h_idx][other_w_idx];
                grad_feat = -2 * vec_ * vec_i * vec_j / std::pow(norm_squared, 2);
              }
              scalar_t grad_out = grad_output[b_idx][ci_idx][cj_idx][h_idx][w_idx];
              omp_set_lock(lock);
              grad_feats[b_idx][c_idx][h_idx][w_idx] += (grad_feat * grad_out);
              grad_feats[b_idx][c_idx][other_h_idx][other_w_idx] += (-grad_feat * grad_out);
              omp_unset_lock(lock);
            }
          }
        }
      }
    }
  }
}

std::vector<at::Tensor> patch_outerprod_cpu_forward(at::Tensor feat_tensor, int kernel_size) {
  auto b_size = feat_tensor.size(0);
  auto c_size = feat_tensor.size(1);
  auto h_size = feat_tensor.size(2);
  auto w_size = feat_tensor.size(3);

  auto output = at::zeros({b_size, c_size, c_size, h_size, w_size}, feat_tensor.options());

  auto world_size = b_size * h_size * w_size;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(feat_tensor.scalar_type(), "patch_outerprod_forward", ([&] {
    at::parallel_for(0, world_size, 0, [&](int64_t start, int64_t end) {
      for (int rank = start; rank < end; rank++) {
        patch_outerprod_forward_kernel(world_size, rank, output.accessor<scalar_t, 5>(),
            feat_tensor.accessor<scalar_t, 4>(), kernel_size, b_size, c_size, h_size, w_size, kernel_size * kernel_size,
            h_size * w_size);
      }
    });
  }));

  return {output};
}

std::vector<at::Tensor> patch_outerprod_cpu_backward(at::Tensor grad_output, at::Tensor feat_tensor, int kernel_size) {
  auto b_size = feat_tensor.size(0);
  auto c_size = feat_tensor.size(1);
  auto h_size = feat_tensor.size(2);
  auto w_size = feat_tensor.size(3);

  auto world_size = b_size * h_size * w_size;

  auto grad_feats = at::zeros({b_size, c_size, h_size, w_size}, feat_tensor.options());

  omp_lock_t lock;
  omp_init_lock(&lock);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(feat_tensor.scalar_type(), "patch_outerprod_backward", ([&] {
    at::parallel_for(0, world_size, 0, [&](int64_t start, int64_t end) {
      for (int rank = start; rank < end; rank++) {
        patch_outerprod_backward_kernel(world_size, rank, grad_output.accessor<scalar_t, 5>(),
            feat_tensor.accessor<scalar_t, 4>(), grad_feats.accessor<scalar_t, 4>(), kernel_size, b_size, c_size,
            h_size, w_size, kernel_size * kernel_size, h_size * w_size, &lock);
      }
    });
  }));

  omp_destroy_lock(&lock);
  return {grad_feats};
}
