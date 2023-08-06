#include <torch/extension.h>
#include <vector>

// TODO: remove this file as it is unused

using namespace torch::indexing;

#define CHECK_CUDA(x) AT_ASSERT(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x)

std::vector<torch::Tensor> sligru_cuda_cell_forward(
  const torch::Tensor& wx,      // [B, H * 2]
  const torch::Tensor& ht_pred, // [B, H]
  const torch::Tensor& u,       // [H * 2, H]
  const torch::Tensor& drop_mask,
  const int normalized_shape,
  const double eps,
  const bool training
) ;

std::vector<torch::Tensor> sligru_cuda_cell_backward(
  const torch::Tensor& grad_out,
  const torch::Tensor& dh_prev,
  const torch::Tensor& zt,
  const torch::Tensor& at,
  const torch::Tensor& drop_mask,
  const torch::Tensor& ht,
  const torch::Tensor& hcand,
  const torch::Tensor& u,
  const torch::Tensor& du_prev,
  const torch::Tensor& recurrent_gate,
  const torch::Tensor& mean,
  const torch::Tensor& rstd,
  const int normalized_shape,
  const bool training
);

std::vector<torch::Tensor> sligru_forward(
  const torch::Tensor& wx,      // [B, H * 2]
  const torch::Tensor& ht_pred, // [B, H]
  const torch::Tensor& u,       // [H * 2, H]
  const torch::Tensor& drop_mask, // [B, H]
  const int normalized_shape,
  const double eps,
  const bool training) {

  return sligru_cuda_cell_forward(
    wx, 
    ht_pred, 
    u, 
    drop_mask, 
    normalized_shape, 
    eps,
    training
  );
}

std::vector<torch::Tensor> sligru_backward(
  const torch::Tensor& grad_out,
  const torch::Tensor& dh_prev,
  const torch::Tensor& zt,
  const torch::Tensor& at,
  const torch::Tensor& drop_mask,
  const torch::Tensor& ht,
  const torch::Tensor& hcand,
  const torch::Tensor& u,
  const torch::Tensor& du_prev,
  const torch::Tensor& recurrent_gate,
  const torch::Tensor& mean,
  const torch::Tensor& rstd,
  const int normalized_shape,
  const bool training) {
  CHECK_INPUT(grad_out);
  CHECK_INPUT(dh_prev);
  CHECK_INPUT(zt);
  CHECK_INPUT(at);
  CHECK_INPUT(drop_mask);
  CHECK_INPUT(ht);
  CHECK_INPUT(hcand);
  CHECK_INPUT(u);
  CHECK_INPUT(du_prev);
  CHECK_INPUT(recurrent_gate);
  CHECK_INPUT(mean);
  CHECK_INPUT(rstd);

  return sligru_cuda_cell_backward(
    grad_out, dh_prev, zt, 
    at, drop_mask, ht,
     hcand, u, du_prev,
     recurrent_gate, mean, rstd,
     normalized_shape, training);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sligru_forward, "SLi-GRU forward (CUDA)");
  m.def("backward", &sligru_backward, "SLi-GRU backward (CUDA)");
}