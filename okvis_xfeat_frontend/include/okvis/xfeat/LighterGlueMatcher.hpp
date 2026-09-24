/**
 * @file okvis/xfeat/LighterGlueMatcher.hpp
 * @brief LighterGlue (LightGlue trained for XFeat 64-D descriptors) pair
 *        matcher on TensorRT — the learned replacement for brute-force
 *        descriptor matching between two images (ADR-0040 stage B).
 *
 * Engine contract (xfeat_lightglue_onnx_mowe/export.py, opset 18):
 *   inputs  "kpts0"/"kpts1" [1,K,2] float — NORMALISED coords, LightGlue
 *           convention: (px - size/2) / (max(W,H)/2); this class normalises
 *           internally from pixel inputs
 *           "desc0"/"desc1" [1,K,64] float — L2-normalised XFeat descriptors
 *   outputs "matches" [M,2] int64 (mutual-NN index pairs), "scores" [M] float
 *           — M is DATA-DEPENDENT, handled via a preallocated
 *           nvinfer1::IOutputAllocator (M <= K always)
 *
 * K is the engine's fixed capacity (2048 for the bench engine); shorter sets
 * are zero-padded and matches touching padding slots are discarded.
 */
#ifndef OKVIS_XFEAT_LIGHTERGLUEMATCHER_HPP_
#define OKVIS_XFEAT_LIGHTERGLUEMATCHER_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace okvis {
namespace xfeat {

struct LighterGlueConfig {
  std::string engine_path;  ///< Serialized .plan; empty → stub (loaded()==false).
  float min_score = 0.10f;  ///< Mutual-NN confidence floor to accept a match.
  int cuda_device = 0;      ///< CUDA device ordinal.
};

/// \brief Result of matching keypoint set A against set B.
struct PairMatches {
  /// Accepted (indexA, indexB) pairs, indices into the caller's sets.
  std::vector<std::pair<std::uint32_t, std::uint32_t>> indices;
  std::vector<float> scores;  ///< Per-pair mutual-NN confidence. Same size.
  std::size_t size() const noexcept { return indices.size(); }
};

/// \brief Owns the LighterGlue TensorRT engine + CUDA stream. Not copyable;
///        movable. NOT thread-safe — serialise match() calls externally.
class LighterGlueMatcher {
 public:
  explicit LighterGlueMatcher(const LighterGlueConfig& cfg);
  ~LighterGlueMatcher();
  LighterGlueMatcher(LighterGlueMatcher&&) noexcept;
  LighterGlueMatcher& operator=(LighterGlueMatcher&&) noexcept;
  LighterGlueMatcher(const LighterGlueMatcher&) = delete;
  LighterGlueMatcher& operator=(const LighterGlueMatcher&) = delete;

  /// True when a real TensorRT engine is loaded (vs. stub mode).
  bool loaded() const noexcept;

  /// The engine's fixed keypoint capacity K (0 in stub mode).
  std::uint32_t capacity() const noexcept;

  /// Match keypoint set A against set B.
  /// kpts*: [n x 2] float PIXEL coords (x, y), contiguous row-major.
  /// desc*: [n x 64] float, L2-normalised, contiguous (XFeat layout — e.g. the
  /// CV_32F descriptor Mat of an okvis Frame). Coordinates are normalised
  /// internally using each image's (width, height). Sets longer than
  /// capacity() are truncated. Returns only matches with both indices in
  /// range and score >= min_score; empty result in stub mode / on error.
  PairMatches match(const float* kptsA, const float* descA, std::size_t nA,
                    std::uint32_t widthA, std::uint32_t heightA,
                    const float* kptsB, const float* descB, std::size_t nB,
                    std::uint32_t widthB, std::uint32_t heightB);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace xfeat
}  // namespace okvis

#endif  // OKVIS_XFEAT_LIGHTERGLUEMATCHER_HPP_
