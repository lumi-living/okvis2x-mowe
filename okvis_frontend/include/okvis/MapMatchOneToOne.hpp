/**
 * @file MapMatchOneToOne.hpp
 * @brief T-0129: one keypoint per landmark in Frontend::matchToMap (float/XFeat path).
 *
 * Upstream lets every keypoint inside the IMU-predicted gate take its best landmark
 * independently. XFeat samples descriptors from a 1/8-resolution map, so keypoints a few
 * px apart carry near-identical descriptors and one landmark collects several keypoints,
 * all but one off by the keypoint spacing (the ~6 px "large reprojection error" and the
 * RANSAC FAIL at inlier ratio ~0.67 under XFeat; out/agent/T-0129/diagnosis.md).
 */
#ifndef INCLUDE_OKVIS_MAPMATCHONETOONE_HPP_
#define INCLUDE_OKVIS_MAPMATCHONETOONE_HPP_

#include <cmath>
#include <unordered_map>
#include <vector>

#include <okvis/FrameTypedefs.hpp>

namespace okvis {

/// \brief Keep, per landmark, only the keypoint with the lowest descriptor distance
///        (ties: the lower keypoint index); the others' lmIds are reset.
/// \return Number of keypoints dropped.
inline size_t keepBestKeypointPerLandmark(std::vector<LandmarkId>& lmIds,
                                          const std::vector<double>& distances) {
  std::unordered_map<uint64_t, size_t> bestK;
  size_t dropped = 0;
  for (size_t k = 0; k < lmIds.size(); ++k) {
    if (!lmIds[k].isInitialised()) continue;
    auto ins = bestK.emplace(lmIds[k].value(), k);
    if (ins.second) continue;
    size_t& j = ins.first->second;
    const size_t drop = distances[k] < distances[j] ? j : k;
    if (drop == j) j = k;
    lmIds[drop] = LandmarkId();
    ++dropped;
  }
  return dropped;
}

/// \brief Lowe ratio test on L2 descriptor distance: drop keypoint k when its best
///        landmark is not clearly better than the best *other* landmark in its gate,
///        i.e. sqrt(d1/d2) >= ratio. Distances are cosine distances of unit rows, so
///        L2 = sqrt(2 d). ratio >= 1 disables the test.
/// \return Number of keypoints dropped.
inline size_t rejectAmbiguousMatches(std::vector<LandmarkId>& lmIds,
                                     const std::vector<double>& distances,
                                     const std::vector<double>& secondDistances,
                                     double ratio) {
  if (ratio >= 1.0) return 0;
  size_t dropped = 0;
  for (size_t k = 0; k < lmIds.size(); ++k) {
    if (!lmIds[k].isInitialised()) continue;
    if (distances[k] >= ratio * ratio * secondDistances[k]) {
      lmIds[k] = LandmarkId();
      ++dropped;
    }
  }
  return dropped;
}

}  // namespace okvis

#endif  // INCLUDE_OKVIS_MAPMATCHONETOONE_HPP_
