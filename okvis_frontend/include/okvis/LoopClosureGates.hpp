/**
 * @file LoopClosureGates.hpp
 * @brief Loop-closure candidate gates that need no TensorRT (Mow-e T-0120):
 *        the Mahalanobis prior gate, the pixel -> RANSAC threshold conversion
 *        and the OpenGV GP3P absolute-pose RANSAC used to verify a candidate.
 *        Pure CPU so the gtest runs under qemu-aarch64 (mowe-nav-kb 06 §aliasing).
 * @author Mow-e
 */

#ifndef INCLUDE_OKVIS_LOOPCLOSUREGATES_HPP_
#define INCLUDE_OKVIS_LOOPCLOSUREGATES_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/kinematics/Transformation.hpp>

namespace okvis {
namespace loopclosure {

/// \brief Drift model of the VIO prior between two keyframes (KB 06 defence #1).
///        OKVIS2 has no cheap marginal covariance on the real-time path (KB 11), so
///        the prior covariance is diagonal: position sigma = floor + drift_frac * path
///        length travelled between the frames (the same heuristic the estimator's
///        drift_percentage_heuristic uses), orientation sigma constant.
struct PriorGateParams {
  double sigma_pos_floor_m = 0.3;
  double drift_frac = 0.0135;
  double sigma_rot_rad = 30.0 * M_PI / 180.0;
};

/// \brief Mahalanobis distance of the candidate's pose relative to the current one
///        under the drift-model prior. Position and orientation parts add in
///        quadrature (6-D diagonal covariance), so the gate `distance <= sigma`
///        treats a 3-sigma position offset and a 3-sigma yaw offset alike.
/// \param T_WS_new  Current frame pose (W = VIO world).
/// \param T_WS_old  Candidate keyframe pose in the same W.
/// \param pathLength_m  Path length travelled between the two frames [m].
inline double priorMahalanobis(const kinematics::Transformation& T_WS_new,
                               const kinematics::Transformation& T_WS_old,
                               double pathLength_m, const PriorGateParams& p) {
  const double sigmaPos = p.sigma_pos_floor_m + p.drift_frac * std::max(0.0, pathLength_m);
  const double dPos = (T_WS_new.r() - T_WS_old.r()).norm() / sigmaPos;
  // Rotation angle between the two attitudes (full 3-axis; fisheye rigs see the
  // same place with pitch/roll changes too, so no yaw-only shortcut).
  const Eigen::Quaterniond dq = T_WS_old.q().inverse() * T_WS_new.q();
  const double angle = 2.0 * std::atan2(dq.vec().norm(), std::abs(dq.w()));
  const double dRot = angle / p.sigma_rot_rad;
  return std::sqrt(dPos * dPos + dRot * dRot);
}

/// \brief RANSAC threshold for FrameAbsolutePoseSacProblem from a pixel tolerance.
///        The SAC problem scores `|e_bearing|^2 / sigmaAngle` with sigmaAngle =
///        sqrt(2) * (0.8 * keypointSize / 12)^2 / fu^2 (LoopclosureNoncentralAbsoluteAdapter)
///        and |e_bearing| ~ e_px / fu, so the threshold is fu-independent:
///        reproj_px^2 / (sqrt(2) * (0.8 * size / 12)^2). Upstream's literal 16 equals
///        ~5.1 px at the default BRISK-ish size 16.
inline double ransacThresholdFromPixels(double reproj_px, double keypointSize_px) {
  const double sigmaPx = 0.8 * keypointSize_px / 12.0;
  return reproj_px * reproj_px / (std::sqrt(2.0) * sigmaPx * sigmaPx);
}

/// \brief Temporal consistency (KB 06 defence #2, SeqSLAM-lite): a geometrically
///        verified candidate is accepted only when the last `required` CONSECUTIVE
///        keyframe queries each verified a candidate, and those candidates lie within
///        `maxGap` database keyframes of each other. A query without a verified
///        candidate (miss()) clears the window, unlike mowe_vpr::SequenceVerifier,
///        whose window only slides — with sparse loop verifications that would let
///        two agreeing hits minutes apart count as consecutive.
class TemporalConsistency {
 public:
  TemporalConsistency(int required, int maxGap) : required_(required), maxGap_(maxGap) {}
  /// \brief Record this query's verified candidate (database index); true = accept.
  bool push(int dbIndex) {
    window_.push_back(dbIndex);
    while (int(window_.size()) > required_) {
      window_.pop_front();
    }
    if (int(window_.size()) < required_) {
      return false;
    }
    int lo = window_.front(), hi = window_.front();
    for (int v : window_) {
      lo = std::min(lo, v);
      hi = std::max(hi, v);
    }
    return hi - lo <= maxGap_;
  }
  /// \brief This query verified nothing: consecutive run broken.
  void miss() { window_.clear(); }
  int required() const { return required_; }

 private:
  int required_, maxGap_;
  std::deque<int> window_;
};

/// \brief GP3P absolute-pose RANSAC of the new multi-frame against 3-D points
///        given in the OLD frame's sensor frame S_old (the way pose-graph frames
///        store them, ViSlamBackend::convertToPoseGraphMst).
/// \param points     landmark id -> homogeneous point in S_old.
/// \param matches    keypoint of frameNew -> landmark id.
/// \param frameNew   The new multi-frame (keypoints, back-projections, T_SC).
/// \param threshold  SAC threshold (see ransacThresholdFromPixels).
/// \param maxIterations RANSAC iterations.
/// \param[out] T_Sold_Snew  Pose of the new sensor frame in S_old.
/// \param[out] inlierMask   Per-correspondence inlier flags, in adapter order.
/// \param[out] camIndices/keypointIndices  Adapter order -> (camera, keypoint) of frameNew.
/// \return Number of inliers (0 when fewer than 7 correspondences).
int ransacAbsolutePose(const AlignedMap<uint64_t, Eigen::Vector4d>& points,
                       const std::map<KeypointIdentifier, uint64_t>& matches,
                       const std::shared_ptr<const MultiFrame>& frameNew,
                       double threshold, int maxIterations,
                       kinematics::Transformation& T_Sold_Snew,
                       std::vector<bool>& inlierMask,
                       std::vector<size_t>& camIndices,
                       std::vector<size_t>& keypointIndices);

}  // namespace loopclosure
}  // namespace okvis

#endif  // INCLUDE_OKVIS_LOOPCLOSUREGATES_HPP_
