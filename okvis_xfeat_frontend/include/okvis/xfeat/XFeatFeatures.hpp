/**
 * @file okvis/xfeat/XFeatFeatures.hpp
 * @brief Output of the XFeat TensorRT frontend: per-stream keypoints +
 *        descriptors, ready to hand to OKVIS.
 *
 * This is the boundary type between the Mow-e camera/inference world and OKVIS.
 * It is deliberately framework-light (Eigen + std) so it can be produced by the
 * TensorRT frontend without dragging OKVIS internals into the inference TU, and
 * consumed by the OKVIS-side adapter that builds an okvis::MultiFrame.
 */
#ifndef OKVIS_XFEAT_XFEATFEATURES_HPP_
#define OKVIS_XFEAT_XFEATFEATURES_HPP_

#include <cstdint>
#include <string>
#include <vector>

namespace okvis {
namespace xfeat {

/// XFeat descriptor dimensionality (verlab/accelerated_features).
inline constexpr std::uint32_t kDescriptorDim = 64;

/// Pixel-coordinate keypoint. Plain POD on purpose — this is a cross-boundary
/// type (inference world ↔ OKVIS), so it pulls in no math library; the OKVIS
/// adapter converts to cv::KeyPoint / Eigen as needed.
struct Keypoint {
  float u = 0.f;
  float v = 0.f;
};

/// \brief Features for a single image plane ("left" | "right" | "color").
struct StreamFeatures {
  std::string stream_id;

  /// Keypoint pixel coordinates (u,v) in the ORIGINAL image frame (i.e. already
  /// mapped back from the network's padded/normalised input). Size == N.
  std::vector<Keypoint> keypoints_px;

  /// Per-keypoint reliability/score from the XFeat heatmap. Size == N.
  std::vector<float> scores;

  /// Row-major [N x kDescriptorDim] L2-normalised float descriptors. These are
  /// FLOAT descriptors (cosine/L2 matching), NOT binary — the OKVIS adapter must
  /// use the float-descriptor matching path, not BRISK Hamming. See README.
  std::vector<float> descriptors;

  std::size_t size() const noexcept { return keypoints_px.size(); }
};

/// \brief All synchronized streams from one capture, with its SOF timestamp.
struct FrameFeatures {
  std::int64_t sequence = 0;
  std::int64_t timestamp_ns = 0;  ///< CLOCK_MONOTONIC SOF (cam/IMU sync key).
  std::vector<StreamFeatures> streams;
};

}  // namespace xfeat
}  // namespace okvis

#endif  // OKVIS_XFEAT_XFEATFEATURES_HPP_
