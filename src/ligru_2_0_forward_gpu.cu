// Copyright 2022 Adel Moumen. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <stdexcept>

#include "blas.h"
#include "device_assert.h"

#include "inline_ops.h"
#include "layer_norm.h"
#include "ligru_2_0.h"

namespace {

template <typename T, haste::v0::ligru_2_0::ActivationFunction<T> activation, bool Training>
__global__ void PointwiseOperations(
    const int batch_dim,
    const int hidden_dim,
    const T *wx,
    const T *uh,
    const T *h,
    T *h_out,
    T *v
) {
  const int row = blockDim.x * blockIdx.x + threadIdx.x;
  const int col = blockDim.y * blockIdx.y + threadIdx.y;

  if (row >= hidden_dim || col >= batch_dim)
    return;

  const int weight_idx = col * (hidden_dim * 2) + row;
  const int output_idx = col * hidden_dim + row;

  const int a_idx = weight_idx + 0 * hidden_dim;
  const int z_idx = weight_idx + 1 * hidden_dim;

  const T z = sigmoid(wx[z_idx] + uh[z_idx]);
  const T a = wx[a_idx] + uh[z_idx];

  const T hcand = activation(a); 

  if (Training) {
    const int base_v_idx = col * (hidden_dim * 3) + row;
    v[base_v_idx + 1 * hidden_dim] = z;
    v[base_v_idx + 0 * hidden_dim] = a;
    v[base_v_idx + 2 * hidden_dim] = hcand;
  }

  T cur_h_value = z * h[output_idx] + (static_cast<T>(1.0) - z) * hcand;
  h_out[output_idx] = cur_h_value;
}
} // anonymous namespace

namespace haste {
namespace v0 {
namespace ligru_2_0 {

template <typename T> struct ForwardPass<T>::private_data {
  bool training;
  int batch_size;
  int input_size;
  int hidden_size;
  int activation;
  cublasHandle_t blas_handle;
  cudaStream_t stream[2];
  cudaEvent_t event;
  cudaStream_t sync_stream;
};

template <typename T>
ForwardPass<T>::ForwardPass(const bool training, const int batch_size,
                            const int input_size, const int hidden_size,
                            const cublasHandle_t &blas_handle,
                            const int activation, const cudaStream_t &stream)
    : data_(new private_data) {
  data_->activation = activation;
  data_->training = training;
  data_->batch_size = batch_size;
  data_->input_size = input_size;
  data_->hidden_size = hidden_size;
  data_->blas_handle = blas_handle;
  data_->sync_stream = stream;
  cudaStreamCreate(&data_->stream[0]);
  cudaStreamCreate(&data_->stream[1]);
  cudaEventCreateWithFlags(&data_->event, cudaEventDisableTiming);
}

template <typename T> ForwardPass<T>::~ForwardPass() {
  if (data_->sync_stream) {
    cudaEventRecord(data_->event, data_->stream[1]);
    cudaStreamWaitEvent(data_->sync_stream, data_->event, 0);
    cudaEventRecord(data_->event, data_->stream[0]);
    cudaStreamWaitEvent(data_->sync_stream, data_->event, 0);
  } else {
    cudaStreamSynchronize(data_->stream[1]);
    cudaStreamSynchronize(data_->stream[0]);
  }
  cudaEventDestroy(data_->event);
  cudaStreamDestroy(data_->stream[1]);
  cudaStreamDestroy(data_->stream[0]);
  delete data_;
}

// returns a PointwiseOperations kernel given a runtime activation_method
template<typename T, bool training>
auto wrap_pointwise_activation(int activation_method)
{
    switch (activation_method)
    {
    case 0: return PointwiseOperations<T, relu, training>;
    case 1: return PointwiseOperations<T, leaky_relu, training>;
    case 2: return PointwiseOperations<T, sin, training>;
    case 3: return PointwiseOperations<T, tanh, training>;
    }

    throw std::invalid_argument("Invalid activation_method passed");
}

// returns a PointwiseOperations kernel given a runtime
// training+activation_method, by dispatching the proper value as a template
template<typename T>
auto wrap_pointwise(bool training, int activation_method)
{
    if (training)
    {
        return wrap_pointwise_activation<T, true>(activation_method);
    }
    else
    {
        return wrap_pointwise_activation<T, false>(activation_method);
    }
} 

template <typename T>
void ForwardPass<T>::IterateInternal(
    const T *u,
    const T *h,
    T *h_out,
    T *v,
    T *tmp_wx,
    T *tmp_uh,
    T *tmp_uh_norm,
    layer_norm::ForwardPass<T> &layer_norm1
) {
  const bool training = data_->training;
  const int batch_size = data_->batch_size;
  const int hidden_size = data_->hidden_size;
  const cublasHandle_t blas_handle = data_->blas_handle;
  const cudaStream_t stream1 = data_->stream[0];
  const cudaEvent_t event = data_->event;

  cublasSetStream(blas_handle, stream1);

  const static auto alpha = static_cast<T>(1.0f);
  const static auto beta = static_cast<T>(0.0f);
  // tmp_uh = 1 * u . h + 0 * tmp_uh aka tmp_uh = u.h
  // gemm is C <- alpha * A * B + beta * C
  blas<T>::gemm(
      blas_handle,
      CUBLAS_OP_N, CUBLAS_OP_N,                 // no transposition
      hidden_size * 2, batch_size, hidden_size, // shape
      &alpha,                                   // alpha == 1.0
      u, hidden_size * 2,                       // A
      h, hidden_size,                           // B
      &beta,                                    // beta == 0.0
      tmp_uh, hidden_size * 2                   // C, used only as a target
  );

  // NOTE: With FP16 inputs & outputs, LN will be performed in FP32 space internally
  layer_norm1.RunPartial(stream1, batch_size, tmp_uh, tmp_uh_norm);

  // Compute launch configuration for pointwise operations kernel.
  const dim3 blockDim(32, 16);
  const dim3 gridDim((hidden_size + blockDim.x - 1) / blockDim.x,
                     (batch_size + blockDim.y - 1) / blockDim.y);

  cudaStreamWaitEvent(stream1, event, 0);

  const auto kernel = wrap_pointwise<T>(training, data_->activation);
  kernel<<<gridDim, blockDim, 0, stream1>>>(
    batch_size, hidden_size, tmp_wx, tmp_uh_norm, h, h_out, v
  );
}

template <typename T>
void ForwardPass<T>::Run(
    const int seq_length,
    T *wx, const T *u, T *h, T *v,
    layer_norm::ForwardPass<T> &layer_norm1,
    T *tmp_uh_norm,
    T *tmp_uh) {

  const blas<void>::set_pointer_mode scoped1(data_->blas_handle);

  const int batch_size = data_->batch_size;
  const int hidden_size = data_->hidden_size;
  const cublasHandle_t blas_handle = data_->blas_handle;

  cudaStream_t save_stream;
  cublasGetStream(blas_handle, &save_stream);

  const int NH = batch_size * hidden_size;

  for (int i = 0; i < seq_length; ++i) {
    IterateInternal(u, h + i * NH, h + (i + 1) * NH, v + i * NH * 3,
                    wx + i * NH * 2, tmp_uh + i * NH * 2, tmp_uh_norm,
                    layer_norm1);
  }

  cublasSetStream(blas_handle, save_stream);
}

template struct ForwardPass<half>;
template struct ForwardPass<float>;
template struct ForwardPass<double>;

} // namespace ligru_2_0
} // namespace v0
} // namespace haste