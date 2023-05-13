#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) AT_ASSERT(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x)

std::vector<torch::Tensor> sligru_cuda_cell_forward(
  const torch::Tensor& wx,      // [B, H * 2]
  const torch::Tensor& ht_pred, // [B, H]
  const torch::Tensor& u,       // [H * 2, H]
  const torch::Tensor& drop_mask,
  int normalized_shape,
  double eps
)

std::vector<torch::Tensor> sligru_cell_forward(
  const torch::Tensor& wx,      // [B, H * 2]
  const torch::Tensor& ht_pred, // [B, H]
  const torch::Tensor& u,       // [H * 2, H]
  const torch::Tensor& drop_mask, // [B, H]
  const int normalized_shape,
  const double eps) {
  return sligru_cuda_cell_forward(wx, ht_pred, u, drop_mask, normalized_shape, eps);
}

std::vector<torch::Tensor> sligru_forward(
  const torch::Tensor& wx,      // [B, T, H * 2]
  const torch::Tensor& ht_pred, // [B, H]
  const torch::Tensor& u,       // [H * 2, H]
  const torch::Tensor& drop_mask, // [B, H]
  const int normalized_shape,
  const double eps) {
  CHECK_INPUT(wx);
  CHECK_INPUT(ht_pred);
  CHECK_INPUT(u);
  CHECK_INPUT(drop_mask);
  return sligru_cell_forward(wx, ht_pred, u, drop_mask, normalized_shape, eps);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sligru_forward, "SLi-GRU forward (CUDA)");
}