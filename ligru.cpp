#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) AT_ASSERT(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x)


std::vector<at::Tensor> ligru_forward(
    at::Tensor wx, 
    at::Tensor u,
    at::Tensor ht
    ) 
{

    CHECK_INPUT(wx);

    const int batch_size = wx.size(0);
    const int seq_length = wx.size(1);
    const int hidden_size = ht.size(1);

    std::cout << batch_size << std::endl;
    std::cout << seq_length << std::endl;
    std::cout << hidden_size << std::endl;

    return {wx};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &ligru_forward, "Li-GRU forward (CUDA)");
}