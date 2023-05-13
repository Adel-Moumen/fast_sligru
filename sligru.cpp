#include <torch/extension.h>
#include <vector>

using namespace torch::indexing;

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
) ;

std::vector<torch::Tensor> sligru_forward(
  const torch::Tensor& wx,      // [B, H * 2]
  const torch::Tensor& ht_pred, // [B, H]
  const torch::Tensor& u,       // [H * 2, H]
  const torch::Tensor& drop_mask, // [B, H]
  const int normalized_shape,
  const double eps) {
  
  // const int batch = wx.size(0);
  // const int time = wx.size(1);
  // const int hidden = ht_pred.size(1);

  // auto ht_out = torch::zeros({batch, time, hidden}, wx.device());

  //std::cout << ht_out.sizes() << std::endl;
  //std::cout << ht_pred.sizes() << std::endl;
  //std::cout << "here" << std::endl;

  // ht_out.index_put_({Slice(), 0}, ht_pred);

  //std::cout << ht_out.index({Slice(), 0}) << std::endl;

  /*
  for (int64_t t = 0; t < time; ++t) {
    //std::cout << "---------------------" << std::endl;
    //std::cout << "t = " << t << std::endl;
    //std::cout << wx.index({Slice(), t-1}) << std::endl;
    //std::cout << ht_out.index({Slice(), t-1}) << std::endl;
    // auto out = sligru_cuda_cell_forward(wx.index({Slice(), t-1}), ht_out.index({Slide(), t-1}), u, drop_mask, normalized_shape, eps);
    // std:: cout << std::get<0>(out) << std::endl;

    if (t == 0) {
      ht_out.index({Slice(), t}) =  sligru_cuda_cell_forward(
        wx.index({Slice(), t}), 
        ht_pred, 
        u, 
        drop_mask, 
        normalized_shape, 
        eps
      )[0];
    }
    else {
      ht_out.index({Slice(), t}) =  sligru_cuda_cell_forward(
        wx.index({Slice(), t}), 
        ht_out.index({Slice(), t-1}), 
        u, 
        drop_mask, 
        normalized_shape, 
        eps
      )[0];
    }
  }
  */

  return sligru_cuda_cell_forward(
    wx, 
    ht_pred, 
    u, 
    drop_mask, 
    normalized_shape, 
    eps
  );

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sligru_forward, "SLi-GRU forward (CUDA)");
}