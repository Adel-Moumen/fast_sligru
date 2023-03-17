#pragma once

#include <cuda_runtime_api.h>

namespace haste {
namespace v0 {
namespace layer_norm {

/// Exposes a typedef ::type depending on the accumulator data type that should
/// be used during the layer normalization. We want to compute LN in at least
/// FP32.
template<typename T>
struct LNormTypeSelector {};

// Support FP64
template<>
struct LNormTypeSelector<double> {
    using type = double;
};

// Support FP32
template<>
struct LNormTypeSelector<float> {
    using type = float;
};

// Convert FP16 to FP32
template<>
struct LNormTypeSelector<half> : LNormTypeSelector<float> {};

template <typename T> class ForwardPass {
public:
  // gamma: [H]
  // beta: [H]
  // cache: [N,2]
  ForwardPass(const int batch_size, const int hidden_size, const T *gamma,
              const T *beta, T *cache);

  // Computes the layer norm of an input tensor `x` over its innermost (fastest
  // changing) dimension. The layer norm is defined as: \(\frac{x-\mu}{\sigma}
  // \gamma + \beta\) where `\gamma` and `\beta` are trainable parameters.
  //
  // x: [N,H]
  // y: [N,H]
  void Run(const cudaStream_t &stream, const T *x, T *y);

  void RunPartial(const cudaStream_t &stream, const int minibatch, const T *x,
                  T *y);

private:
  const int batch_size_;
  const int hidden_size_;
  const T *gamma_;
  const T *beta_;
  T *cache_;
  int partial_;
};

template <typename T> class BackwardPass {
public:
  BackwardPass(const int batch_size, const int hidden_size, const T *gamma,
               const T *beta, const T *x, T *dgamma, T *dbeta, T *cache);

  void Run(const cudaStream_t &stream, const T *dy, T *dx);

  void RunPartial(const cudaStream_t &stream, const int minibatch, const T *dy,
                  T *dx);

private:
  const int batch_size_;
  const int hidden_size_;
  const T *gamma_;
  const T *beta_;
  const T *x_;
  T *dgamma_;
  T *dbeta_;
  T *cache_;
  int partial_;
};

} // namespace layer_norm
} // namespace v0
} // namespace haste