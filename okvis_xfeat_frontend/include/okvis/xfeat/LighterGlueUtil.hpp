/**
 * @file okvis/xfeat/LighterGlueUtil.hpp
 * @brief CPU-only helpers around the LighterGlue split-matcher engine
 *        (xfeat_lightglue_onnx_mowe/export.py, T-0109): top-K selection by
 *        score, staging into the fixed [1,K] slots with the score-derived
 *        validity mask (KB 03 §padding), and decoding the fixed-shape
 *        matches0/mscores0 outputs back to caller indices. Header-only so the
 *        gtest runs on host, qemu and device without CUDA. // T-0114
 */
#ifndef OKVIS_XFEAT_LIGHTERGLUEUTIL_HPP_
#define OKVIS_XFEAT_LIGHTERGLUEUTIL_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

namespace okvis {
namespace xfeat {

/// Sentinel written into unused slots' scores: the engine masks `score <= 0`
/// keys out of every attention and out of the assignment (export.py).
inline constexpr float kLighterGluePadScore = -1.f;

/// Indices of the `k` highest-scoring rows (descending), or all rows when
/// n <= k. Ties keep the lower index first (deterministic).
inline std::vector<std::uint32_t> top_k_by_score(const float* scores,
                                                 std::size_t n, std::size_t k) {
  std::vector<std::uint32_t> order(n);
  std::iota(order.begin(), order.end(), std::uint32_t(0));
  auto better = [scores](std::uint32_t a, std::uint32_t b) {
    return scores[a] > scores[b] || (scores[a] == scores[b] && a < b);
  };
  if (n > k) {
    std::partial_sort(order.begin(), order.begin() + std::ptrdiff_t(k), order.end(), better);
    order.resize(k);
  } else {
    std::sort(order.begin(), order.end(), better);
  }
  return order;
}

/// One engine side, staged host-side: exactly K slots.
struct StagedSet {
  std::vector<float> kpts_norm;          ///< [K x 2], LightGlue-normalised; 0 in unused slots
  std::vector<float> scores;             ///< [K]; kLighterGluePadScore in unused slots
  std::vector<std::uint32_t> src_index;  ///< used slot i -> caller row (size == used)
  std::size_t used = 0;
  std::size_t capacity() const noexcept { return scores.size(); }
  /// Slots the engine will mask (score <= 0), incl. caller rows with score <= 0.
  std::size_t masked_slots() const noexcept {
    return std::size_t(std::count_if(scores.begin(), scores.end(), [](float s) { return s <= 0.f; }));
  }
};

/// Stage the top-K (by score) of a caller set into `out`. Pixel coords are
/// normalised the way kornia's normalize_keypoints does:
///   (xy - size/2) / (max(W,H)/2)   (export.py contract).
/// Descriptors are NOT gathered here (they stay on the caller's side; use
/// out.src_index to copy rows) so the device upload can gather directly.
inline void stage_set(const float* kpts_px, const float* scores, std::size_t n,
                      std::uint32_t width, std::uint32_t height,
                      std::size_t K, StagedSet& out) {
  out.kpts_norm.assign(K * 2, 0.f);
  out.scores.assign(K, kLighterGluePadScore);
  out.src_index = top_k_by_score(scores, n, K);
  out.used = out.src_index.size();
  const float shiftX = 0.5f * float(width);
  const float shiftY = 0.5f * float(height);
  const float invScale = 2.0f / float(std::max(width, height));
  for (std::size_t i = 0; i < out.used; ++i) {
    const std::uint32_t r = out.src_index[i];
    out.kpts_norm[i * 2] = (kpts_px[r * 2] - shiftX) * invScale;
    out.kpts_norm[i * 2 + 1] = (kpts_px[r * 2 + 1] - shiftY) * invScale;
    out.scores[i] = scores[r];
  }
}

/// Decoded (callerIndexA, callerIndexB) pairs + engine confidences.
struct DecodedMatches {
  std::vector<std::pair<std::uint32_t, std::uint32_t>> indices;
  std::vector<float> scores;
  std::size_t size() const noexcept { return indices.size(); }
};

/// Decode the engine's matches0[K] (index into set B or -1) / mscores0[K]
/// back to caller indices. Slots outside the staged range, targets outside B's
/// staged range, -1 entries and scores below `min_score` are dropped. The
/// engine already masks padding, so out-of-range targets never happen with a
/// correct engine; the guard keeps garbage from becoming a match regardless.
inline DecodedMatches decode_matches(const std::int32_t* matches0,
                                     const float* mscores0,
                                     const StagedSet& a, const StagedSet& b,
                                     float min_score) {
  DecodedMatches out;
  out.indices.reserve(a.used);
  out.scores.reserve(a.used);
  for (std::size_t i = 0; i < a.used; ++i) {
    const std::int32_t j = matches0[i];
    if (j < 0 || std::size_t(j) >= b.used) continue;
    if (a.scores[i] <= 0.f || b.scores[std::size_t(j)] <= 0.f) continue;  // masked slot
    const float s = mscores0[i];
    if (s < min_score) continue;
    out.indices.emplace_back(a.src_index[i], b.src_index[std::size_t(j)]);
    out.scores.push_back(s);
  }
  return out;
}

}  // namespace xfeat
}  // namespace okvis

#endif  // OKVIS_XFEAT_LIGHTERGLUEUTIL_HPP_
