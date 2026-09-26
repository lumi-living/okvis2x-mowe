/**
 * @file okvis/xfeat/LighterGlueMatcher.hpp
 * @brief LighterGlue (LightGlue trained for XFeat 64-D descriptors) pair
 *        matcher on TensorRT — used where wide-baseline context matters:
 *        stereo fallback when mutual-NN yields too few matches and loop
 *        verification (mowe-nav-kb 02 decision table; ADR-0040 issue 4).
 *
 * Engine contract (xfeat_lightglue_onnx_mowe/export.py, T-0109, static shapes):
 *   inputs  "kpts0"/"kpts1"     [1,K,2]  float — kornia normalize_keypoints:
 *                                        (px - size/2) / (max(W,H)/2)
 *           "desc0"/"desc1"     [1,K,64] float — L2-normalised XFeat rows
 *           "scores0"/"scores1" [1,K]    float — validity = score > 0; padded
 *                                        slots (<= 0) are masked out of every
 *                                        attention and of the assignment
 *   outputs "matches0" [1,K] int32 (index into set 1 or -1; mutual-NN and the
 *           filter threshold applied in-graph), "mscores0" [1,K] float.
 *
 * K is the engine's fixed capacity (512 for the stereo/loop engine). Longer
 * sets are cut to the top-K by score; shorter ones are padded with score -1.
 * Runs on a greatest-priority non-blocking CUDA stream like the extractor
 * (KB 03 §streams). // T-0114
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
  float min_score = 0.10f;  ///< Confidence floor on mscores0 to accept a match.
  int cuda_device = 0;      ///< CUDA device ordinal.
};

/// \brief Result of matching keypoint set A against set B.
struct PairMatches {
  /// Accepted (indexA, indexB) pairs, indices into the caller's sets.
  std::vector<std::pair<std::uint32_t, std::uint32_t>> indices;
  std::vector<float> scores;  ///< Per-pair engine confidence. Same size.
  std::uint32_t staged_a = 0; ///< Rows of A actually fed (min(nA, capacity)).
  std::uint32_t staged_b = 0; ///< Rows of B actually fed.
  std::uint32_t masked_a = 0; ///< Slots of A the engine masked (padding + score<=0).
  std::uint32_t masked_b = 0; ///< Slots of B the engine masked.
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

  /// Engine I/O tensors are exactly the export.py contract above.
  bool io_names_match() const noexcept;

  /// Match keypoint set A against set B.
  /// kpts*: [n x 2] float PIXEL coords (x, y), contiguous row-major.
  /// desc*: [n x 64] float, L2-normalised, contiguous (XFeat layout — e.g. the
  /// CV_32F descriptor Mat of an okvis Frame).
  /// scores*: [n] XFeat scores; rows with score <= 0 are masked by the engine.
  /// Coordinates are normalised internally from each image's (width, height).
  /// Sets longer than capacity() keep their top-capacity rows by score.
  /// Returns only matches between staged, unmasked rows with confidence >=
  /// min_score; empty result in stub mode / on error.
  PairMatches match(const float* kptsA, const float* descA, const float* scoresA,
                    std::size_t nA, std::uint32_t widthA, std::uint32_t heightA,
                    const float* kptsB, const float* descB, const float* scoresB,
                    std::size_t nB, std::uint32_t widthB, std::uint32_t heightB);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace xfeat
}  // namespace okvis

#endif  // OKVIS_XFEAT_LIGHTERGLUEMATCHER_HPP_
