/**
 * @file preprocess.cu
 * @brief CUDA preprocessing for the XFeat TensorRT frontend: pitched uint8 mono
 *        → resized float NCHW plane, raw 0..255 (see preprocess.h for the why).
 */
#include "preprocess.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace {

// Pixel-centre aligned bilinear resize (src = (dst + 0.5) * scale - 0.5, edges
// clamped) — the inverse lives in FrameFeaturesUtil.hpp::to_full_res, keep
// them in sync.
__global__ void resize_u8_to_nchw_f32(const std::uint8_t* __restrict__ src,
                                      int src_pitch, int src_w, int src_h,
                                      float* __restrict__ dst, int dst_w,
                                      int dst_h, float sx, float sy) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dst_w || y >= dst_h) return;
  float fx = (x + 0.5f) * sx - 0.5f;
  float fy = (y + 0.5f) * sy - 0.5f;
  fx = fmaxf(fx, 0.f);
  fy = fmaxf(fy, 0.f);
  const int x0 = min(int(fx), src_w - 1);
  const int y0 = min(int(fy), src_h - 1);
  const int x1 = min(x0 + 1, src_w - 1);
  const int y1 = min(y0 + 1, src_h - 1);
  const float wx = fx - x0;
  const float wy = fy - y0;
  const float p00 = src[y0 * src_pitch + x0];
  const float p01 = src[y0 * src_pitch + x1];
  const float p10 = src[y1 * src_pitch + x0];
  const float p11 = src[y1 * src_pitch + x1];
  const float top = p00 + (p01 - p00) * wx;
  const float bot = p10 + (p11 - p10) * wx;
  // Raw value — NO normalisation (XFeat InstanceNorm handles scale).
  dst[y * dst_w + x] = top + (bot - top) * wy;
}

}  // namespace

extern "C" void xfeat_preprocess_resize_u8(const void* src, int src_pitch,
                                           int src_w, int src_h, float* dst,
                                           int dst_w, int dst_h, void* stream) {
  const dim3 block(16, 16);
  const dim3 grid((dst_w + block.x - 1) / block.x,
                  (dst_h + block.y - 1) / block.y);
  resize_u8_to_nchw_f32<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const std::uint8_t*>(src), src_pitch, src_w, src_h, dst,
      dst_w, dst_h, float(src_w) / float(dst_w), float(src_h) / float(dst_h));
}
