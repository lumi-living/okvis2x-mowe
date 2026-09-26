/**
 * @file test_loop_closure_gates.cpp
 * @brief Mow-e T-0120: loop-closure candidate gates without TensorRT — the
 *        Mahalanobis prior gate, temporal consistency, the pixel→RANSAC
 *        threshold conversion, and GP3P RANSAC on a synthetic stereo rig with
 *        known outliers through the real verification code
 *        (okvis::loopclosure::ransacAbsolutePose). CPU only: qemu-aarch64 + device.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <random>
#include <vector>

#include <okvis/LoopClosureGates.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/NCameraSystem.hpp>
#include <okvis/cameras/PinholeCamera.hpp>

using okvis::kinematics::Transformation;
using okvis::loopclosure::PriorGateParams;
using okvis::loopclosure::priorMahalanobis;
using okvis::loopclosure::TemporalConsistency;
using okvis::loopclosure::ransacThresholdFromPixels;

namespace {

Transformation pose(double x, double y, double z, double yawDeg) {
  return Transformation(Eigen::Vector3d(x, y, z),
                        Eigen::Quaterniond(Eigen::AngleAxisd(yawDeg * M_PI / 180.0,
                                                             Eigen::Vector3d::UnitZ())));
}

// Two equidistant-fisheye cameras (the TUM-VI / OV9281 case), 11 cm baseline.
okvis::cameras::NCameraSystem makeStereo() {
  okvis::cameras::NCameraSystem cams;
  for (int i = 0; i < 2; ++i) {
    std::shared_ptr<const Transformation> T_SC(
        new Transformation(Eigen::Vector3d(0.11 * i, 0, 0), Eigen::Quaterniond::Identity()));
    std::shared_ptr<const okvis::cameras::CameraBase> cam(
        new okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion>(
            512, 512, 190.0, 190.0, 256.0, 256.0,
            okvis::cameras::EquidistantDistortion(0.0035, 0.0007, -0.002, 0.0002)));
    cams.addCamera(T_SC, cam, okvis::cameras::NCameraSystem::DistortionType::Equidistant,
                   /*computeOverlaps=*/false);
  }
  return cams;
}

}  // namespace

TEST(LoopClosureGates, PriorGateMahalanobis) {
  PriorGateParams p;
  p.sigma_pos_floor_m = 0.5;
  p.drift_frac = 0.0;
  p.sigma_rot_rad = 30.0 * M_PI / 180.0;
  const Transformation T0 = pose(1, 2, 0, 10);
  EXPECT_NEAR(priorMahalanobis(T0, T0, 0.0, p), 0.0, 1e-12);
  // 1 m offset at sigma 0.5 -> 2 sigma; 90 deg yaw at sigma 30 deg -> 3 sigma
  EXPECT_NEAR(priorMahalanobis(pose(2, 2, 0, 10), T0, 0.0, p), 2.0, 1e-9);
  EXPECT_NEAR(priorMahalanobis(pose(1, 2, 0, 100), T0, 0.0, p), 3.0, 1e-9);
  // both together add in quadrature
  EXPECT_NEAR(priorMahalanobis(pose(2, 2, 0, 100), T0, 0.0, p), std::sqrt(13.0), 1e-9);
  // the drift model widens the position sigma with path length: 1 m offset after
  // 100 m at 1.35 %/m -> sigma 0.5 + 1.35 = 1.85 m
  p.drift_frac = 0.0135;
  EXPECT_NEAR(priorMahalanobis(pose(2, 2, 0, 10), T0, 100.0, p), 1.0 / 1.85, 1e-9);
  // a candidate on the other side of a 5 m room facing the opposite wall is
  // rejected at 3 sigma even after 100 m of path; the same place revisited with
  // 10 cm drift and 20 deg of yaw drift passes.
  EXPECT_GT(priorMahalanobis(pose(6, 2, 0, 190), T0, 100.0, p), 3.0);
  EXPECT_LT(priorMahalanobis(pose(1.1, 2, 0, 30), T0, 100.0, p), 3.0);
}

TEST(LoopClosureGates, TemporalConsistency) {
  TemporalConsistency tc(2, 5);
  EXPECT_FALSE(tc.push(10));       // first verification: window not full
  EXPECT_TRUE(tc.push(12));        // consecutive and within 5 keyframes
  EXPECT_TRUE(tc.push(14));        // keeps agreeing while the loop is revisited
  EXPECT_FALSE(tc.push(80));       // jumps to a different place: disagree
  EXPECT_TRUE(tc.push(83));        // ...then agrees with the new place
  tc.miss();                       // a keyframe query with nothing verified breaks the run
  EXPECT_FALSE(tc.push(84));
  EXPECT_TRUE(tc.push(84));
  TemporalConsistency one(1, 0);   // required=1 accepts immediately
  EXPECT_TRUE(one.push(3));
}

TEST(LoopClosureGates, RansacThresholdFromPixels) {
  // upstream's literal threshold 16 corresponds to ~5.07 px at keypoint size 16
  const double t16 = ransacThresholdFromPixels(5.07395, 16.0);
  EXPECT_NEAR(t16, 16.0, 0.001);
  // 2 px -> 2.486 at size 16; grows with the pixel tolerance squared
  EXPECT_NEAR(ransacThresholdFromPixels(2.0, 16.0), 2.486, 0.005);
  EXPECT_NEAR(ransacThresholdFromPixels(4.0, 16.0) / ransacThresholdFromPixels(2.0, 16.0), 4.0, 1e-9);
}

TEST(LoopClosureGates, SyntheticPnpWithKnownOutliers) {
  std::mt19937 rng(7);
  std::srand(7);  // opengv's RANSAC samples with rand()
  const okvis::cameras::NCameraSystem cams = makeStereo();
  std::shared_ptr<okvis::MultiFrame> frame(new okvis::MultiFrame(cams, okvis::Time(1.0), 42));

  // ground truth: the new sensor frame seen from the old one
  const Transformation T_Sold_Snew_gt(Eigen::Vector3d(0.35, -0.2, 0.1),
                                      Eigen::Quaterniond(Eigen::AngleAxisd(
                                          12.0 * M_PI / 180.0, Eigen::Vector3d(0.2, 1.0, 0.3).normalized())));
  std::uniform_real_distribution<double> ux(-2.0, 2.0), uz(2.0, 6.0), upx(0.0, 512.0);
  std::normal_distribution<double> noise(0.0, 0.3);
  okvis::AlignedMap<uint64_t, Eigen::Vector4d> points;
  std::map<okvis::KeypointIdentifier, uint64_t> matches;
  std::vector<std::vector<bool>> isOutlier(2);
  uint64_t lmId = 1;
  int numInliersTrue = 0, numOutliersTrue = 0;
  for (size_t im = 0; im < 2; ++im) {
    std::vector<cv::KeyPoint> kps;
    const Transformation T_Sold_C = T_Sold_Snew_gt * (*cams.T_SC(im));
    for (int n = 0; n < 60; ++n) {
      // landmark in the OLD sensor frame, in front of the new camera
      const Eigen::Vector3d p_C(ux(rng), ux(rng), uz(rng));
      const Eigen::Vector4d hp_Sold = T_Sold_C * Eigen::Vector4d(p_C[0], p_C[1], p_C[2], 1.0);
      Eigen::Vector2d px;
      if (cams.cameraGeometry(im)->project(p_C, &px) !=
          okvis::cameras::ProjectionStatus::Successful) {
        continue;
      }
      const bool outlier = (n % 3 == 0);  // every third correspondence is wrong
      if (outlier) {
        px = Eigen::Vector2d(upx(rng), upx(rng));
        ++numOutliersTrue;
      } else {
        px += Eigen::Vector2d(noise(rng), noise(rng));
        ++numInliersTrue;
      }
      isOutlier[im].push_back(outlier);
      kps.emplace_back(float(px[0]), float(px[1]), 16.f, -1.f, 1.f);
      points[lmId] = hp_Sold;
      matches[okvis::KeypointIdentifier(frame->id(), im, kps.size() - 1)] = lmId;
      ++lmId;
    }
    frame->resetKeypoints(im, kps);
    frame->computeBackProjections(im);
  }
  ASSERT_GT(numInliersTrue, 40);
  ASSERT_GT(numOutliersTrue, 20);

  Transformation T_Sold_Snew;
  std::vector<bool> inlierMask;
  std::vector<size_t> camIdx, kpIdx;
  const int numInliers = okvis::loopclosure::ransacAbsolutePose(
      points, matches, frame, ransacThresholdFromPixels(2.0, 16.0), 200, T_Sold_Snew,
      inlierMask, camIdx, kpIdx);
  EXPECT_GE(numInliers, int(0.9 * numInliersTrue));
  EXPECT_LE(numInliers, numInliersTrue);
  // no synthetic outlier survives as an inlier
  int leakedOutliers = 0;
  for (size_t k = 0; k < inlierMask.size(); ++k) {
    if (inlierMask[k] && isOutlier[camIdx[k]][kpIdx[k]]) {
      ++leakedOutliers;
    }
  }
  EXPECT_EQ(leakedOutliers, 0);
  // recovered pose: this is the raw GP3P minimal-sample model (Frontend refines it
  // with Ceres afterwards), so with 0.3 px noise at 2-6 m depth a few cm / ~1 deg
  // is normal; an outlier-contaminated model would be off by metres / tens of deg.
  const Transformation dT = T_Sold_Snew_gt.inverse() * T_Sold_Snew;
  EXPECT_LT(dT.r().norm(), 0.10);
  EXPECT_LT(2.0 * std::atan2(dT.q().vec().norm(), std::abs(dT.q().w())) * 180.0 / M_PI, 2.0);
}

TEST(LoopClosureGates, PnpTooFewCorrespondencesIsRejected) {
  const okvis::cameras::NCameraSystem cams = makeStereo();
  std::shared_ptr<okvis::MultiFrame> frame(new okvis::MultiFrame(cams, okvis::Time(1.0), 1));
  frame->resetKeypoints(0, std::vector<cv::KeyPoint>(3, cv::KeyPoint(100.f, 100.f, 16.f)));
  frame->computeBackProjections(0);
  okvis::AlignedMap<uint64_t, Eigen::Vector4d> points;
  std::map<okvis::KeypointIdentifier, uint64_t> matches;
  for (size_t k = 0; k < 3; ++k) {
    points[k + 1] = Eigen::Vector4d(0.1 * k, 0, 3, 1);
    matches[okvis::KeypointIdentifier(1, 0, k)] = k + 1;
  }
  Transformation T;
  std::vector<bool> mask;
  std::vector<size_t> c, kp;
  EXPECT_EQ(okvis::loopclosure::ransacAbsolutePose(points, matches, frame, 2.5, 50, T, mask, c, kp), 0);
}
