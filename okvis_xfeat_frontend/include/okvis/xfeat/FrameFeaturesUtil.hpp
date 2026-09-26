/**
 * @file okvis/xfeat/FrameFeaturesUtil.hpp
 * @brief CPU-only helpers that turn raw XFeat engine tensors into
 *        StreamFeatures: padding strip, full-resolution coordinate scale-back,
 *        unit-norm check. Header-only so the gtest runs on host, qemu and
 *        device without CUDA. // T-0111, mowe-nav-kb 02 §procedures
 */
#ifndef OKVIS_XFEAT_FRAMEFEATURESUTIL_HPP_
#define OKVIS_XFEAT_FRAMEFEATURESUTIL_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "okvis/xfeat/XFeatFeatures.hpp"

namespace okvis {
namespace xfeat {

/// Full-resolution ↔ engine-resolution mapping. The preprocess kernel samples
/// the source at pixel centres (OpenCV INTER_LINEAR convention):
///   src = (dst + 0.5) * s - 0.5,  s = full / engine   (per axis).
/// For the OV9281 1280x800 → 640x384 engine: sx = 2.0, sy = 800/384 = 2.0833.
/// The inverse below therefore places an engine keypoint at the centre of the
/// full-res pixel block it was sampled from.
struct ResizeScale {
  float sx = 1.f;
  float sy = 1.f;
};

inline ResizeScale resize_scale(std::uint32_t full_w, std::uint32_t full_h,
                                std::uint32_t eng_w, std::uint32_t eng_h) {
  return {float(full_w) / float(eng_w), float(full_h) / float(eng_h)};
}

inline Keypoint to_full_res(std::int32_t x_eng, std::int32_t y_eng,
                            ResizeScale s) {
  return {(float(x_eng) + 0.5f) * s.sx - 0.5f,
          (float(y_eng) + 0.5f) * s.sy - 0.5f};
}

/// Build `out` from one batch slot of the engine outputs (export.py contract:
/// keypoints int32 [K,2] (x,y) in engine pixels, scores [K] with -1 in padding
/// slots, descriptors [K,64] L2-normalised). Rows with score < 0 (padding) or
/// below `score_threshold` are dropped. Returns the number of padding rows
/// (score < 0) that were stripped; `out.padding_rows` gets the same value.
inline std::uint32_t assemble_stream(const std::int32_t* kp_xy,
                                     const float* scores, const float* desc,
                                     std::uint32_t K, ResizeScale scale,
                                     float score_threshold,
                                     StreamFeatures& out) {
  out.keypoints_px.clear();
  out.scores.clear();
  out.descriptors.clear();
  out.keypoints_px.reserve(K);
  out.scores.reserve(K);
  out.descriptors.reserve(std::size_t(K) * kDescriptorDim);
  std::uint32_t padding = 0;
  for (std::uint32_t i = 0; i < K; ++i) {
    const float s = scores[i];
    if (s < 0.f) {  // -1 sentinel: padding slot (KB 03 §static shapes)
      ++padding;
      continue;
    }
    if (s < score_threshold) continue;
    out.keypoints_px.push_back(to_full_res(kp_xy[2 * i], kp_xy[2 * i + 1], scale));
    out.scores.push_back(s);
    const float* d = desc + std::size_t(i) * kDescriptorDim;
    out.descriptors.insert(out.descriptors.end(), d, d + kDescriptorDim);
  }
  out.padding_rows = padding;
  return padding;
}

/// max_i | ‖desc_i‖₂ − 1 | over the [N x 64] rows; 0 for an empty set.
inline float max_unit_norm_deviation(const std::vector<float>& desc) {
  float worst = 0.f;
  for (std::size_t r = 0; r + kDescriptorDim <= desc.size(); r += kDescriptorDim) {
    double acc = 0.0;
    for (std::uint32_t k = 0; k < kDescriptorDim; ++k) acc += double(desc[r + k]) * desc[r + k];
    worst = std::max(worst, float(std::fabs(std::sqrt(acc) - 1.0)));
  }
  return worst;
}

}  // namespace xfeat
}  // namespace okvis

#endif  // OKVIS_XFEAT_FRAMEFEATURESUTIL_HPP_
