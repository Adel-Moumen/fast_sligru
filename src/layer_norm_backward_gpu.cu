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

#include "inline_ops.h"
#include "layer_norm.h"

namespace {

template <typename T, typename Acc, bool ApplyBeta>
__global__ void LayerNormGrad(const int batch_size, const int hidden_size,
                              const T *gamma, const T *x, const T *dy,
                              T *dgamma, T *dbeta, T *dx, T *cache) {
  const int batch = blockDim.x * blockIdx.x + threadIdx.x;
  if (batch >= batch_size)
    return;

  extern __shared__ int shared_var[];
  auto *shared = reinterpret_cast<Acc *>(shared_var);
  const int index = threadIdx.y;
  const int stride = blockDim.y;
  const int batch_idx = batch * hidden_size;
  const int batch_block_idx = threadIdx.x * stride * 3;

  const Acc mean = cache[batch * 2 + 0];
  const Acc invstd = cache[batch * 2 + 1];

  auto dsigma_tmp = static_cast<Acc>(0.0);
  auto dmu1_tmp = static_cast<Acc>(0.0);
  auto dmu2_tmp = static_cast<Acc>(0.0);
  for (int i = index; i < hidden_size; i += stride) {
    const auto cur_dy = static_cast<Acc>(dy[batch_idx + i]);
    const auto centered_x = static_cast<Acc>(x[batch_idx + i]) - mean;
    // const T z = centered_x * invstd;

    const Acc db = cur_dy;
    dsigma_tmp += centered_x * db;
    dmu1_tmp += centered_x;
    dmu2_tmp += db;
  }
  shared[batch_block_idx + index * 3 + 0] = dsigma_tmp;
  shared[batch_block_idx + index * 3 + 1] = dmu1_tmp;
  shared[batch_block_idx + index * 3 + 2] = dmu2_tmp;
  __syncthreads();

  for (int s = stride / 2; s > 0; s >>= 1) {
    if (index < s) {
      shared[batch_block_idx + index * 3 + 0] +=
          shared[batch_block_idx + (index + s) * 3 + 0];
      shared[batch_block_idx + index * 3 + 1] +=
          shared[batch_block_idx + (index + s) * 3 + 1];
      shared[batch_block_idx + index * 3 + 2] +=
          shared[batch_block_idx + (index + s) * 3 + 2];
    }
    __syncthreads();
  }

  const auto dsigma = static_cast<Acc>(-0.5) * shared[batch_block_idx + 0] * invstd *
                   invstd * invstd;
  const auto dmu = (static_cast<Acc>(-2.0) * shared[batch_block_idx + 1] * dsigma /
                 static_cast<Acc>(hidden_size)) -
                (shared[batch_block_idx + 2] * invstd);

  for (int i = index; i < hidden_size; i += stride) {
    const auto cur_dy = static_cast<Acc>(dy[batch_idx + i]);
    const auto centered_x = static_cast<Acc>(x[batch_idx + i]) - mean;

    const auto db = cur_dy;
    dx[batch_idx + i] =
        (static_cast<Acc>(2.0) * centered_x * dsigma / static_cast<Acc>(hidden_size)) +
        (invstd * db) + (dmu / static_cast<Acc>(hidden_size));
  }
}

} // anonymous namespace

namespace haste {
namespace v0 {
namespace layer_norm {

template <typename T>
BackwardPass<T>::BackwardPass(const int batch_size, const int hidden_size,
                              const T *gamma, const T *beta, const T *x,
                              T *dgamma, T *dbeta, T *cache)
    : batch_size_(batch_size), hidden_size_(hidden_size), gamma_(gamma),
      beta_(beta), x_(x), dgamma_(dgamma), dbeta_(dbeta), cache_(cache),
      partial_(batch_size) {}

template <typename T>
void BackwardPass<T>::Run(const cudaStream_t &stream, const T *dy, T *dx) {
  RunPartial(stream, batch_size_, dy, dx);
}

template <typename T>
void BackwardPass<T>::RunPartial(const cudaStream_t &stream,
                                 const int minibatch, const T *dy, T *dx) {
  assert(partial_ - minibatch >= 0);

  dim3 blockDim(4, 256);
  dim3 gridDim;
  gridDim.x = (minibatch + blockDim.x - 1) / blockDim.x;
  using Acc = typename LNormTypeSelector<T>::type;
  const int shared_mem_size = sizeof(Acc) * blockDim.x * blockDim.y * 3;

  if (beta_ && dbeta_) {
    LayerNormGrad<T, Acc, true><<<gridDim, blockDim, shared_mem_size, stream>>>(
        minibatch, hidden_size_, gamma_,
        x_ + (partial_ - minibatch) * hidden_size_, dy, dgamma_, dbeta_, dx,
        cache_ + (partial_ - minibatch) * 2);
  } else {
    LayerNormGrad<T, Acc, false><<<gridDim, blockDim, shared_mem_size, stream>>>(
        minibatch, hidden_size_, gamma_,
        x_ + (partial_ - minibatch) * hidden_size_, dy, dgamma_, nullptr, dx,
        cache_ + (partial_ - minibatch) * 2);
  }

  partial_ -= minibatch;
}

template class BackwardPass<half>;
template class BackwardPass<float>;
template class BackwardPass<double>;

} // namespace layer_norm
} // namespace v0
} // namespace haste