/**
 * @file LoopClosureGates.cpp
 * @brief GP3P RANSAC extracted from Frontend::verifyRecognisedPlace so the
 *        synthetic-outlier gtest exercises the real verification code (T-0120).
 */

#include <okvis/LoopClosureGates.hpp>

#include <opengv/absolute_pose/LoopclosureNoncentralAbsoluteAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/FrameAbsolutePoseSacProblem.hpp>

namespace okvis {
namespace loopclosure {

int ransacAbsolutePose(const AlignedMap<uint64_t, Eigen::Vector4d>& points,
                       const std::map<KeypointIdentifier, uint64_t>& matches,
                       const std::shared_ptr<const MultiFrame>& frameNew,
                       double threshold, int maxIterations,
                       kinematics::Transformation& T_Sold_Snew,
                       std::vector<bool>& inlierMask,
                       std::vector<size_t>& camIndices,
                       std::vector<size_t>& keypointIndices) {
  opengv::absolute_pose::LoopclosureNoncentralAbsoluteAdapter adapter(
      points, matches, frameNew->cameraSystem(), frameNew);
  typedef opengv::sac_problems::absolute_pose::FrameAbsolutePoseSacProblem<
      opengv::absolute_pose::LoopclosureNoncentralAbsoluteAdapter>
      LoopclosureAbsoluteModel;
  const size_t n = adapter.getNumberCorrespondences();
  inlierMask.assign(n, false);
  camIndices.resize(n);
  keypointIndices.resize(n);
  for (size_t k = 0; k < n; ++k) {
    camIndices[k] = size_t(adapter.camIndex(k));
    keypointIndices[k] = size_t(adapter.keypointIndex(k));
  }
  if (n < 7) {
    return 0;
  }
  opengv::sac::Ransac<LoopclosureAbsoluteModel> ransac;
  std::shared_ptr<LoopclosureAbsoluteModel> problem(
      new LoopclosureAbsoluteModel(adapter, LoopclosureAbsoluteModel::Algorithm::GP3P));
  ransac.sac_model_ = problem;
  ransac.threshold_ = threshold;
  ransac.max_iterations_ = maxIterations;
  ransac.computeModel(0);
  for (int idx : ransac.inliers_) {
    inlierMask.at(size_t(idx)) = true;
  }
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.topLeftCorner<3, 4>() = ransac.model_coefficients_;
  T_Sold_Snew = kinematics::Transformation(T);
  return int(ransac.inliers_.size());
}

}  // namespace loopclosure
}  // namespace okvis
