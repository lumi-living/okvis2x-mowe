/**
 * @file mowe_map/Relocaliser.hpp
 * @brief Boot / kidnapped relocalisation against a saved .mowemap (Mow-e T-0121,
 *        mowe-nav-kb 06 §boot, ADR-0015 under ADR-0042). Lives in the fork because it
 *        needs LighterGlue (TensorRT) and OpenGV; the container itself is the ROS-free
 *        onboard/localization/mowe_map package.
 *
 *        Per frame: VPR top-K over the map keyframes -> LighterGlue old->new per camera
 *        (proposals landing on initialised landmarks) -> GP3P RANSAC + non-linear
 *        refinement -> verified if inliers >= min_inliers and ratio >= min_inlier_ratio.
 *        Confident only when consecutive_required consecutive frames each verified a
 *        candidate AND the recovered T_WS of consecutive frames agree within agree_pos_m /
 *        agree_rot_deg (poses of DIFFERENT frames, so the tolerance must cover the motion
 *        between them: TUM-VI handheld moves 0.27 m / 12 deg median per 0.25 s; the mower
 *        at <= 1 m/s far less). A miss clears the window. The in-session loop closure's
 *        keyframe-INDEX gap is deliberately not used here: a lawn map revisits the same
 *        place at far-apart indices, so consecutive true matches jump indices (T-0121 run 1).
 */
#ifndef INCLUDE_MOWE_MAP_RELOCALISER_HPP_
#define INCLUDE_MOWE_MAP_RELOCALISER_HPP_

#include <deque>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <mowe_map/mowemap.hpp>
#include <mowe_vpr/retrieval.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/cameras/NCameraSystem.hpp>
#include <okvis/kinematics/Transformation.hpp>

namespace okvis { namespace xfeat { class LighterGlueMatcher; } }

namespace mowe_map {

struct RelocaliserParams {
  int top_k = 3;
  double score_min = 0.40;        ///< whitened-VLAD cosine gate (T-0119 calibration)
  int min_inliers = 20;
  double reproj_px = 2.0;
  double min_inlier_ratio = 0.4;
  int consecutive_required = 2;
  double agree_pos_m = 1.0;       ///< consecutive recovered poses must agree this closely (incl. motion)
  double agree_rot_deg = 60.0;    ///< catches flipped / mirrored solutions, not handheld rotation
  double keypoint_size_px = 16.0; ///< frontend xfeat.keypoint_size (RANSAC threshold scale)
  int ransac_iterations = 50;
};

struct Relocalisation {
  bool verified = false;   ///< this frame alone passed retrieval + geometry
  bool confident = false;  ///< KB 06 §boot (a) inliers, (b) VPR score, (c) temporal agreement
  okvis::kinematics::Transformation T_WS;  ///< sensor pose in the map's W (valid if verified)
  Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Zero();
  int keyframe_index = -1;
  float vpr_score = 0.f;    ///< score of the verified candidate
  float top_score = 0.f;    ///< best retrieval score this frame, gated or not (diagnostics)
  int top_keyframe = -1;
  int inliers = 0;
  int correspondences = 0;  ///< LighterGlue proposals on initialised landmarks
  int candidates = 0;       ///< candidates above score_min that were geometrically tried
};

class Relocaliser {
 public:
  /// Builds the retrieval database and per-keyframe MultiFrames from the map (this is
  /// the "load into the VPR database + landmark structures" cost; time it outside).
  Relocaliser(const Map& map, const okvis::cameras::NCameraSystem& cameraSystem,
              okvis::xfeat::LighterGlueMatcher& lighterGlue, const RelocaliserParams& params);

  /// \param frame   Stereo frame with XFeat keypoints/descriptors and back-projections set.
  /// \param vprDesc Whitened VLAD of the left image (mowe_vpr::describe), unit norm.
  Relocalisation push(const std::shared_ptr<const okvis::MultiFrame>& frame,
                      const Eigen::VectorXf& vprDesc);
  void reset() { window_.clear(); }
  size_t size() const { return frames_.size(); }
  const okvis::kinematics::Transformation& keyframePose(size_t i) const { return T_WS_.at(i); }
  const RelocaliserParams& params() const { return params_; }

 private:
  bool verify(const okvis::MultiFrame& oldFrame, const std::shared_ptr<const okvis::MultiFrame>& newFrame,
              okvis::kinematics::Transformation& T_Sold_Snew, int& inliers, int& correspondences);

  const okvis::cameras::NCameraSystem& cameraSystem_;
  okvis::xfeat::LighterGlueMatcher& lighterGlue_;
  RelocaliserParams params_;
  mowe_vpr::Database db_;
  std::vector<std::shared_ptr<okvis::MultiFrame>> frames_;
  std::vector<okvis::kinematics::Transformation> T_WS_;
  struct Hit { int keyframe; okvis::kinematics::Transformation T_WS; };
  std::deque<Hit> window_;
};

}  // namespace mowe_map

#endif  // INCLUDE_MOWE_MAP_RELOCALISER_HPP_
