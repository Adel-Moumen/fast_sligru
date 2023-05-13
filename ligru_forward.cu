
#include "ATen/ATen.h"
#include "ATen/AccumulateType.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/DeviceUtils.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

#include "type_shim.h"

template<typename T> __device__ __forceinline__
T sigmoid(const T x) {
  return static_cast<T>(1.0) / (static_cast<T>(1.0) + exp(-x));
}

template<typename T> __device__ __forceinline__
T relu(const T x) {
  return (x > static_cast<T>(0.0f)) ? x : static_cast<T>(0.0f);
}




template<typename T> __global__
void cuApplyCellForward(
    const int time_step,
    const int batch_size,
    const int hidden_size,
    const T* gates,
    T* ht_out
  )
{
    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    const int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row >= hidden_size || col >= batch_size)
        return;


    const int gate_idx = col * 2 * hidden_size + row;
    const double a = relu(gates[gate_idx + 0 * hidden_size]);
    const double z = sigmoid(gates[gate_idx + 1 * hidden_size]);

    const int base_idx = col * hidden_size + row;

    ht_out[base_idx] = static_cast<T>(time_step);

}

template<typename T>
void HostApplyCellForward(
    const int batch_size,
    const int time_step,
    const int hidden_size,
    const T* gates,
    const T* ht,
    T* ht_out
  )
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const dim3 blockDim(32, 16);
    const dim3 gridDim(
        (hidden_size + blockDim.x - 1) / blockDim.x,
        (batch_size + blockDim.y - 1) / blockDim.y);

    const int NH = batch_size * hidden_size;
    cuApplyCellForward<T><<<gridDim, blockDim, 0, stream>>>(
        time_step,
        batch_size,
        hidden_size,
        gates, 
        ht_out + time_step * NH
    );
}

std::vector<at::Tensor>  ligru_cuda_forward(
    int batch_size, 
    int hidden_size, 
    at::Tensor ht_out,
    at::Tensor wx,
    at::Tensor u,
    at::Tensor ht 
) {    
    using namespace at;



    DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
        ht.scalar_type(), ht_out.scalar_type(),"ligru_forward_cuda",
        using accscalar_t = at::acc_type<scalar_t_in, true>;
        
        for(int t = 0; t < ht_out.size(0); ++t) {
            std::cout << "t: " << t << std::endl;
            std::cout << "wx: " << wx.select(1, t).sizes() << std::endl;
            std::cout << "u: " << u.t().sizes() << std::endl;
            std::cout << "ht: " << ht.sizes() << std::endl;
        
            auto gates = wx.select(1, t) + at::matmul(ht, u.t()); 
            HostApplyCellForward<scalar_t_in>(
                batch_size,
                t,
                hidden_size,
                gates.DATA_PTR<scalar_t_in>(),
                ht.DATA_PTR<scalar_t_in>(),
                ht_out.DATA_PTR<scalar_t_in>()
        );
        
       }
    );

    return {wx};
}