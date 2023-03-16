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

#include <cassert>

#include <cuda_fp16.h>

#include "layer_norm.h"

namespace {

template <typename T, typename Acc, bool ApplyBeta>
__global__ void LayerNorm(const int batch_size, const int hidden_size,
                          const T *gamma, const T *beta, const T *x, T *y,
                          T *cache) {
  const int batch = blockDim.x * blockIdx.x + threadIdx.x;
  if (batch >= batch_size)
    return;

  extern __shared__ int shared_var[];
  auto *shared = reinterpret_cast<Acc *>(shared_var);
  const int index = threadIdx.y;
  const int stride = blockDim.y;
  const int batch_idx = batch * hidden_size;
  const int batch_block_idx = threadIdx.x * stride;

  auto sum = static_cast<Acc>(0.0);
  for (int i = index; i < hidden_size; i += stride)
    sum += static_cast<Acc>(x[batch_idx + i]);
  shared[batch_block_idx + index] = sum;
  __syncthreads();

  for (int s = stride / 2; s > 0; s >>= 1) {
    if (index < s)
      shared[batch_block_idx + index] += shared[batch_block_idx + index + s];
    __syncthreads();
  }

  const Acc mean = static_cast<Acc>(shared[batch_block_idx]) / static_cast<Acc>(hidden_size);
  __syncthreads();

  // Reduce squared difference
  auto sumsq = static_cast<Acc>(0.0);
  for (int i = index; i < hidden_size; i += stride) {
    const Acc diff = static_cast<Acc>(x[batch_idx + i]) - mean;
    sumsq += diff * diff;
  }
  shared[batch_block_idx + index] = sumsq;
  __syncthreads();

  for (int s = stride / 2; s > 0; s >>= 1) {
    if (index < s)
      shared[batch_block_idx + index] += shared[batch_block_idx + index + s];
    __syncthreads();
  }

  const Acc invstd =
      rsqrt(shared[batch_block_idx] / static_cast<Acc>(hidden_size) + static_cast<Acc>(1e-5));

  for (int i = index; i < hidden_size; i += stride) {
    auto out = (static_cast<Acc>(x[batch_idx + i]) - mean) * invstd;
    if (ApplyBeta) {
        out += static_cast<Acc>(beta[i]);
    }
    y[batch_idx + i] = static_cast<T>(out);
  }

  cache[batch * 2 + 0] = mean;
  cache[batch * 2 + 1] = invstd;
}

} // anonymous namespace

namespace haste {
namespace v0 {
namespace layer_norm {

template <typename T>
ForwardPass<T>::ForwardPass(const int batch_size, const int hidden_size,
                            const T *gamma, const T *beta, T *cache)
    : batch_size_(batch_size), hidden_size_(hidden_size), gamma_(gamma),
      beta_(beta), cache_(cache), partial_(0) {}

template <typename T>
void ForwardPass<T>::Run(const cudaStream_t &stream, const T *x, T *y) {
  RunPartial(stream, batch_size_, x, y);
}

template <typename T>
void ForwardPass<T>::RunPartial(const cudaStream_t &stream, const int minibatch,
                                const T *x, T *y) {
  assert(partial_ + minibatch <= batch_size_);

  using Acc = typename LNormTypeSelector<T>::type;

  dim3 blockDim(4, 256);
  dim3 gridDim;
  gridDim.x = (minibatch + blockDim.x - 1) / blockDim.x;
  const int shared_mem_size = sizeof(Acc) * blockDim.x * blockDim.y;

  LayerNorm<T, Acc, false><<<gridDim, blockDim, shared_mem_size, stream>>>(
      minibatch, hidden_size_, nullptr, nullptr, x, y, cache_ + partial_ * 2);

  partial_ += minibatch;
}

template class ForwardPass<half>;
template class ForwardPass<float>;
template class ForwardPass<double>;

} // namespace layer_norm
} // namespace v0
} // namespace haste