#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

#define IDX2(a, b, b_size) ((a)*b_size) + (b)
#define IDX3(a, b, c, bc_size, c_size) ((a)*bc_size) + (IDX2(b, c, c_size))
#define IDX4(a, b, c, d, bcd_size, cd_size, d_size) ((a)*bcd_size) + (IDX3(b, c, d, cd_size, d_size))

#define DISPATCH_CPU_OR_CUDA(device, cpu_fn, cuda_fn, ...) \
  if (device.is_cuda()) {                                  \
    return cuda_fn(__VA_ARGS__);                           \
  } else {                                                 \
    return cpu_fn(__VA_ARGS__);                            \
  }

#define CHECK_SAME_DEVICE(x, y) AT_ASSERTM(x.device() == y.device(), "Tensors should be on the same device!")
