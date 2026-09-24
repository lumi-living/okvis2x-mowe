/**
 * @file preprocess.cu
 * @brief CUDA preprocessing for the XFeat TensorRT frontend.
 *
 * Single-plane uint8 → float NCHW, raw 0..255 (see preprocess.h for the why).
 * Reads the (possibly pitched) device buffer the camera handed us — either the
 * NvBufSurface→CUDA mapped pointer (zero-copy) or a host-uploaded staging
 * buffer — and writes the contiguous engine input binding.
 */
#include "preprocess.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace {

__global__ void mono_u8_to_nchw_f32(const std::uint8_t* __restrict__ src,
                                    int src_pitch, float* __restrict__ dst,
                                    int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  // Raw value, just widened to float — NO normalisation (XFeat InstanceNorm).
  dst[y * width + x] = static_cast<float>(src[y * src_pitch + x]);
}

}  // namespace

extern "C" void xfeat_preprocess_mono_u8(const void* src, int src_pitch,
                                         float* dst, int width, int height,
                                         void* stream) {
  const dim3 block(16, 16);
  const dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);
  mono_u8_to_nchw_f32<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const std::uint8_t*>(src), src_pitch, dst, width, height);
}
