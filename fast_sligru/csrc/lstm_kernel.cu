#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>


namespace {

template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
  return 1.0 / (1.0 + exp(-z));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_sigmoid(scalar_t z) {
  return (1.0 - z) * z;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t relu(scalar_t z) {
  return ((z > static_cast<scalar_t>(0.0f) ) ? z : static_cast<scalar_t>(0.0f));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_relu(scalar_t z) {
  return (z > 0.0) ? 1.0 : 0.0;
}


template <typename scalar_t>
__global__ void lstm_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> it,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> ft,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> gt,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> ot,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> ht_pred,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> ct_pred,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> drop_mask,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input_gate,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> cell_gate,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> forget_gate,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output_gate,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> ht,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> ct,
    const bool training) {
  //batch index
  const int b = blockIdx.y;
  // column index
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h < it.size(1)){
    input_gate[b][h] = sigmoid(it[b][h]);
    cell_gate[b][h] = tanh(gt[b][h]);
    forget_gate[b][h] = sigmoid(ft[b][h]);
    output_gate[b][h] = sigmoid(ot[b][h]);

    ct[b][h] = input_gate[b][h] * cell_gate[b][h] + forget_gate[b][h] * ct_pred[b][h];
    ht[b][h] = output_gate[b][h] * relu(ct[b][h]);
    
    if (training) {
      ht[b][h] *= drop_mask[b][h];
    }
    
  }
}

template <typename scalar_t>
__global__ void lstm_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> grad_out,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> dh_prev,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> zt,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> at,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> ht,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> hcand,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> drop_mask,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> dat,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> dzt,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> grad_dh_prev,
    const bool training) {
  //batch index
  const int b = blockIdx.y;
  // column index
  const int h = blockIdx.x * blockDim.x + threadIdx.x;

  if (h < dat.size(1)){
    auto dh = grad_out[b][h]  + dh_prev[b][h];

    auto tmp = (1. - zt[b][h]) * dh;
    
    dat[b][h] = d_relu(at[b][h]) * tmp;

    if (training) {
      dat[b][h] *= drop_mask[b][h]; 
    }

    dzt[b][h] = (ht[b][h] - hcand[b][h]) * tmp * zt[b][h];
    
    grad_dh_prev[b][h] = dh * zt[b][h];
  }
}

} // namespace



std::vector<torch::Tensor> lstm_cuda_cell_forward(
  const torch::Tensor& wx,      // [B, H * 2]
  const torch::Tensor& ht_pred, // [B, H]
  const torch::Tensor& ct_pred, // [B, H]
  const torch::Tensor& u,       // [H * 2, H]
  const torch::Tensor& drop_mask,
  const bool training 
) {


  const int batch_size = wx.size(0);
  const int hidden_size = ht_pred.size(1);

  auto recurrent_gate = ht_pred.mm(u.t());

  auto gates = wx + recurrent_gate;
  auto chunked_gates = gates.chunk(4, 1);

  auto it = chunked_gates[0];
  auto ft = chunked_gates[1];
  auto gt = chunked_gates[2]; 
  auto ot = chunked_gates[3];

  const int threads = 1024;
  const dim3 blocks((hidden_size + threads - 1) / threads, batch_size);

  auto options = torch::TensorOptions().dtype(wx.dtype()).device(
      torch::kCUDA, wx.device().index());
  

  auto input_gate = torch::empty({batch_size, hidden_size}, options);
  auto forget_gate = torch::empty({batch_size, hidden_size}, options);
  auto output_gate = torch::empty({batch_size, hidden_size}, options);
  auto cell_gate = torch::empty({batch_size, hidden_size}, options);
  auto ht = torch::empty({batch_size, hidden_size}, options);
  auto ct = torch::empty({batch_size, hidden_size}, options);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    wx.type(), "lstm_forward_cuda", ([&] {
    lstm_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        it.packed_accessor32<scalar_t,2, torch::RestrictPtrTraits>(),
        ft.packed_accessor32<scalar_t,2, torch::RestrictPtrTraits>(),
        gt.packed_accessor32<scalar_t,2, torch::RestrictPtrTraits>(),
        ot.packed_accessor32<scalar_t,2, torch::RestrictPtrTraits>(),
        ht_pred.packed_accessor32<scalar_t,2, torch::RestrictPtrTraits>(),
        ct_pred.packed_accessor32<scalar_t,2, torch::RestrictPtrTraits>(),
        drop_mask.packed_accessor32<scalar_t,2, torch::RestrictPtrTraits>(),
        input_gate.packed_accessor32<scalar_t,2, torch::RestrictPtrTraits>(),
        cell_gate.packed_accessor32<scalar_t,2, torch::RestrictPtrTraits>(),
        forget_gate.packed_accessor32<scalar_t,2, torch::RestrictPtrTraits>(),
        output_gate.packed_accessor32<scalar_t,2, torch::RestrictPtrTraits>(),
        ht.packed_accessor32<scalar_t,2, torch::RestrictPtrTraits>(),
        ct.packed_accessor32<scalar_t,2, torch::RestrictPtrTraits>(),
        training);
  }));

  return {ht, ct, input_gate, cell_gate, forget_gate, output_gate, recurrent_gate};
}


std::vector<torch::Tensor> lstm_cuda_cell_backward(
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
  const bool training
) {

  const auto batch_size = dh_prev.size(0);
  const auto hidden_size = dh_prev.size(1);

  const int threads = 1024;
  const dim3 blocks((hidden_size + threads - 1) / threads, batch_size);

  auto options = torch::TensorOptions().dtype(grad_out.dtype()).device(
      torch::kCUDA, grad_out.device().index());

  auto dat = torch::zeros({batch_size, hidden_size}, options);
  auto dzt = torch::zeros({batch_size, hidden_size}, options);
  auto grad_dh_prev = torch::zeros({batch_size, hidden_size}, options);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    dat.type(), "lstm_backward_cuda", ([&] {
    lstm_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        grad_out.packed_accessor32<scalar_t,2, torch::RestrictPtrTraits>(),
        dh_prev.packed_accessor32<scalar_t,2, torch::RestrictPtrTraits>(),
        zt.packed_accessor32<scalar_t,2, torch::RestrictPtrTraits>(),
        at.packed_accessor32<scalar_t,2, torch::RestrictPtrTraits>(),
        ht.packed_accessor32<scalar_t,2, torch::RestrictPtrTraits>(),
        hcand.packed_accessor32<scalar_t,2, torch::RestrictPtrTraits>(),
        drop_mask.packed_accessor32<scalar_t,2, torch::RestrictPtrTraits>(),
        dat.packed_accessor32<scalar_t,2, torch::RestrictPtrTraits>(),
        dzt.packed_accessor32<scalar_t,2, torch::RestrictPtrTraits>(),
        grad_dh_prev.packed_accessor32<scalar_t,2, torch::RestrictPtrTraits>(),
        training);
  }));

    auto dwx = torch::cat({dat, dzt}, /*dim=*/1);
    auto grad_grad_dh_prev = at::addmm(grad_dh_prev, dwx, u); 
    auto du = at::addmm(du_prev, dwx.t(), ht);
    return {dwx, grad_grad_dh_prev, du};
}
