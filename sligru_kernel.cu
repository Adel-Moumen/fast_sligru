#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>


namespace {
  // tanh(zt * 0.5) * 0.5 + 0.5;
template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
  return 1.0 / (1.0 + exp(-z));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t relu(scalar_t z) {
  return fmaxf(0.0, z);
}


template <typename scalar_t>
__global__ void sligru_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> at,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> zt,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> ht_pred,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> drop_mask,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> update_gate,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> hcand,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> ht) {
  //batch index
  const int b = blockIdx.y;
  // column index
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h < at.size(1)){
    update_gate[b][h] = sigmoid(zt[b][h]);
    hcand[b][h] = relu(at[b][h]) * drop_mask[b][h];
    ht[b][h] = ht_pred[b][h] * update_gate[b][h] + (1 - update_gate[b][h]) * hcand[b][h];
    
  }
}

} // namespace



std::vector<torch::Tensor> sligru_cuda_cell_forward(
  const torch::Tensor& wx,      // [B, H * 2]
  const torch::Tensor& ht_pred, // [B, H]
  const torch::Tensor& u,       // [H * 2, H]
  const torch::Tensor& drop_mask,
  int normalized_shape,
  double eps
) {


  const int batch_size = wx.size(0);
  const int hidden_size = ht_pred.size(1);

  auto recurrent_gate = ht_pred.mm(u.t());
  auto normalized_recurrent_gates = torch::native_layer_norm(
      recurrent_gate, 
      normalized_shape, 
      torch::Tensor(), 
      torch::Tensor(),  
      eps
    );

  auto normalized_recurrent_input = std::get<0>(normalized_recurrent_gates);
  auto mean = std::get<1>(normalized_recurrent_gates);
  auto rstd = std::get<2>(normalized_recurrent_gates);


  auto gates = wx + normalized_recurrent_input;
  auto chunked_gates = gates.chunk(2, 1);

  auto at = chunked_gates[0];
  auto zt = chunked_gates[1];

  const int threads = 1024;
  const dim3 blocks((hidden_size + threads - 1) / threads, batch_size);

  auto options = torch::TensorOptions().dtype(wx.dtype()).device(
      torch::kCUDA, wx.device().index());

  auto hcand = torch::empty({batch_size, hidden_size}, options);
  auto update_gate = torch::empty({batch_size, hidden_size}, options);
  auto ht = torch::empty({batch_size, hidden_size}, options);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    wx.type(), "sligru_forward_cuda", ([&] {
    sligru_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        at.packed_accessor32<scalar_t,2, torch::RestrictPtrTraits>(),
        zt.packed_accessor32<scalar_t,2, torch::RestrictPtrTraits>(),
        ht_pred.packed_accessor32<scalar_t,2, torch::RestrictPtrTraits>(),
        drop_mask.packed_accessor32<scalar_t,2, torch::RestrictPtrTraits>(),
        update_gate.packed_accessor32<scalar_t,2, torch::RestrictPtrTraits>(),
        hcand.packed_accessor32<scalar_t,2, torch::RestrictPtrTraits>(),
        ht.packed_accessor32<scalar_t,2, torch::RestrictPtrTraits>());
  }));

  return {ht, hcand, update_gate, at, recurrent_gate, mean, rstd};
}