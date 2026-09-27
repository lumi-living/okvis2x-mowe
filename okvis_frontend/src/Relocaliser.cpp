/**
 * @file Relocaliser.cpp
 * @brief Boot relocalisation against a .mowemap (Mow-e T-0121). See the header.
 */
#include <mowe_map/Relocaliser.hpp>

#include <cmath>
#include <map>

#include <glog/logging.h>
#include <okvis/DescriptorDistance.hpp>
#include <okvis/LoopClosureGates.hpp>
#include <okvis/xfeat/LighterGlueMatcher.hpp>

namespace mowe_map {

Relocaliser::Relocaliser(const Map& map, const okvis::cameras::NCameraSystem& cameraSystem,
                         okvis::xfeat::LighterGlueMatcher& lighterGlue, const RelocaliserParams& params)
    : cameraSystem_(cameraSystem), lighterGlue_(lighterGlue), params_(params) {
  CHECK_EQ(map.header.num_cameras, cameraSystem.numCameras()) << "map / rig camera count";
  frames_.reserve(map.keyframes.size());
  T_WS_.reserve(map.keyframes.size());
  for (const Keyframe& kf : map.keyframes) {
    auto mf = std::make_shared<okvis::MultiFrame>(cameraSystem, okvis::Time().fromNSec(uint64_t(kf.t_ns)), kf.id);
    for (size_t im = 0; im < kf.cameras.size(); ++im) {
      const auto& cam = kf.cameras[im];
      std::vector<cv::KeyPoint> kps(cam.size());
      cv::Mat descriptors(int(cam.size()), okvis::kFloatDescriptorDim, CV_32FC1);
      for (size_t k = 0; k < cam.size(); ++k) {
        kps[k] = cv::KeyPoint(cam[k].x, cam[k].y, cam[k].size, -1.f, cam[k].response);
        std::memcpy(descriptors.ptr<float>(int(k)), cam[k].descriptor.data(), sizeof(float) * okvis::kFloatDescriptorDim);
      }
      mf->resetKeypoints(im, kps);
      mf->resetDescriptors(im, descriptors);
      for (size_t k = 0; k < cam.size(); ++k) {
        mf->setLandmarkId(im, k, cam[k].landmark_id);
        mf->setLandmark(im, k, cam[k].hp_S, cam[k].initialised != 0);
      }
    }
    db_.add(int(frames_.size()), kf.vpr);
    frames_.push_back(mf);
    T_WS_.emplace_back(kf.r_WS, kf.q_WS);
  }
}

// old -> new LighterGlue proposals per camera, kept when they land on an initialised
// landmark of the old (map) keyframe; then GP3P RANSAC + refinement (LoopClosureGates).
// Mirrors Frontend::verifyRecognisedPlace's VPR path without an Estimator.
bool Relocaliser::verify(const okvis::MultiFrame& oldFrame, const std::shared_ptr<const okvis::MultiFrame>& newFrame,
                         okvis::kinematics::Transformation& T_Sold_Snew, int& inliers, int& correspondences) {
  okvis::AlignedMap<uint64_t, Eigen::Vector4d> points;
  std::map<okvis::KeypointIdentifier, uint64_t> matches;
  for (size_t im = 0; im < cameraSystem_.numCameras(); ++im) {
    const size_t nA = oldFrame.numKeypoints(im), nB = newFrame->numKeypoints(im);
    if (!nA || !nB) continue;
    std::vector<float> ka(nA * 2), kb(nB * 2), sa(nA), sb(nB);
    cv::KeyPoint kp;
    for (size_t k = 0; k < nA; ++k) {
      oldFrame.getCvKeypoint(im, k, kp);
      ka[2 * k] = kp.pt.x; ka[2 * k + 1] = kp.pt.y; sa[k] = kp.response;
    }
    for (size_t k = 0; k < nB; ++k) {
      newFrame->getCvKeypoint(im, k, kp);
      kb[2 * k] = kp.pt.x; kb[2 * k + 1] = kp.pt.y; sb[k] = kp.response;
    }
    const auto m = lighterGlue_.match(
        ka.data(), reinterpret_cast<const float*>(oldFrame.keypointDescriptor(im, 0)), sa.data(), nA,
        uint32_t(oldFrame.geometry(im)->imageWidth()), uint32_t(oldFrame.geometry(im)->imageHeight()),
        kb.data(), reinterpret_cast<const float*>(newFrame->keypointDescriptor(im, 0)), sb.data(), nB,
        uint32_t(newFrame->geometry(im)->imageWidth()), uint32_t(newFrame->geometry(im)->imageHeight()));
    for (size_t i = 0; i < m.size(); ++i) {
      const size_t kOld = m.indices[i].first, kNew = m.indices[i].second;
      const uint64_t lmId = oldFrame.landmarkId(im, kOld);
      Eigen::Vector4d hp;
      bool init = false;
      oldFrame.getLandmark(im, kOld, hp, init);
      if (!lmId || !init || hp.norm() < 1e-12) continue;
      const okvis::KeypointIdentifier kid(newFrame->id(), im, kNew);
      if (matches.count(kid)) continue;
      points[lmId] = hp;
      matches[kid] = lmId;
    }
  }
  correspondences = int(matches.size());
  inliers = 0;
  if (correspondences < std::max(params_.min_inliers, 8)) return false;
  std::vector<bool> mask;
  std::vector<size_t> cams, kps;
  inliers = okvis::loopclosure::ransacAbsolutePose(
      points, matches, newFrame,
      okvis::loopclosure::ransacThresholdFromPixels(params_.reproj_px, params_.keypoint_size_px),
      params_.ransac_iterations, T_Sold_Snew, mask, cams, kps, /*refine=*/true);
  const double ratio = mask.empty() ? 0.0 : double(inliers) / double(mask.size());
  return inliers >= params_.min_inliers && ratio >= params_.min_inlier_ratio;
}

Relocalisation Relocaliser::push(const std::shared_ptr<const okvis::MultiFrame>& frame,
                                 const Eigen::VectorXf& vprDesc) {
  Relocalisation out;
  const auto candidates = db_.query(vprDesc, params_.top_k);
  if (!candidates.empty()) {
    out.top_score = candidates.front().score;
    out.top_keyframe = candidates.front().id;
  }
  for (const mowe_vpr::Candidate& c : candidates) {
    if (c.score < float(params_.score_min)) break;  // sorted descending
    ++out.candidates;
    okvis::kinematics::Transformation T_Sold_Snew;
    int inliers = 0, corr = 0;
    if (!verify(*frames_.at(size_t(c.id)), frame, T_Sold_Snew, inliers, corr)) continue;
    out.verified = true;
    out.keyframe_index = c.id;
    out.vpr_score = c.score;
    out.inliers = inliers;
    out.correspondences = corr;
    out.T_WS = T_WS_.at(size_t(c.id)) * T_Sold_Snew;
    break;
  }
  if (!out.verified) {
    window_.clear();  // one miss breaks the consecutive run (KB 06 §boot (c))
    return out;
  }
  window_.push_back({out.keyframe_index, out.T_WS});
  while (int(window_.size()) > params_.consecutive_required) window_.pop_front();
  if (int(window_.size()) == params_.consecutive_required) {
    bool agree = true;
    for (size_t i = 1; i < window_.size(); ++i) {  // consecutive pairs
      const okvis::kinematics::Transformation d = window_[i - 1].T_WS.inverse() * window_[i].T_WS;
      const double ang = 2.0 * std::atan2(d.q().vec().norm(), std::abs(d.q().w())) * 180.0 / M_PI;
      agree = agree && d.r().norm() <= params_.agree_pos_m && ang <= params_.agree_rot_deg;
    }
    out.confident = agree;
  }
  // ponytail: diagonal covariance from the inlier count (0.10 m / 2 deg at min_inliers,
  // shrinking with sqrt(n)); the refined problem's 6x6 information would be the upgrade.
  const double s = std::sqrt(double(params_.min_inliers) / double(std::max(out.inliers, 1)));
  const double sp = 0.10 * s, sr = 2.0 * M_PI / 180.0 * s;
  out.covariance.diagonal() << sp * sp, sp * sp, sp * sp, sr * sr, sr * sr, sr * sr;
  return out;
}

}  // namespace mowe_map
