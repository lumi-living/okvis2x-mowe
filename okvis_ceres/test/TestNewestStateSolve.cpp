/**
 * mow-e (T-0141): the realtime solve ThreadedSlam runs on non-keyframes when
 * estimator_parameters.realtime_cheap_on_nonkeyframes is on — optimiseRealtimeGraph with
 * onlyNewestState — must leave every landmark and every older state bit-identical and move only
 * the newest state (pose, speed, biases); the keyframe (full window) solve moves landmarks.
 */

#include <gtest/gtest.h>
#include <okvis/ViSlamBackend.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>

namespace {
typedef okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion> Camera;

struct Snapshot {
  std::map<uint64_t, Eigen::Vector4d> landmarks;
  std::map<uint64_t, Eigen::Matrix<double, 16, 1>> states;  // r, q (xyzw), speed and biases
};

Snapshot snapshot(const okvis::ViSlamBackend& backend, const std::vector<okvis::StateId>& ids) {
  Snapshot s;
  okvis::MapPoints landmarks;
  backend.getLandmarks(landmarks);
  for (const auto& lm : landmarks) s.landmarks[lm.first.value()] = lm.second.point;
  for (auto id : ids) {
    Eigen::Matrix<double, 16, 1> v;
    v.head<3>() = backend.pose(id).r();
    v.segment<4>(3) = backend.pose(id).q().coeffs();
    v.tail<9>() = backend.speedAndBias(id);
    s.states[id.value()] = v;
  }
  return s;
}
}  // namespace

TEST(okvisTestSuite, NewestStateSolveMovesOnlyNewestState) {
  okvis::ImuParameters imu;
  imu.a0.setZero();
  imu.g0.setZero();  // no default initialiser (mowe-nav-kb 11, T-0125 trap)
  imu.s_a.setOnes();
  imu.g = 9.81;
  imu.a_max = 1000.0;
  imu.g_max = 1000.0;
  imu.sigma_g_c = 6.0e-4;
  imu.sigma_a_c = 2.0e-3;
  imu.sigma_gw_c = 3.0e-6;
  imu.sigma_aw_c = 2.0e-5;
  imu.sigma_bg = 0.01;
  imu.sigma_ba = 0.01;
  imu.use = true;

  const double DURATION = 3.0, DT = 0.01;
  std::srand(1);
  okvis::ImuMeasurementDeque imuMeasurements;
  const okvis::Time t0(100.0);
  for (size_t i = 0; i <= DURATION / DT + 10; ++i) {
    imuMeasurements.push_back(okvis::ImuMeasurement(
        t0 + okvis::Duration(DT * i),
        okvis::ImuSensorReadings(Eigen::Vector3d::Random() * 1e-4,
                                 Eigen::Vector3d(0, 0, imu.g) + Eigen::Vector3d::Random() * 1e-3)));
  }

  std::shared_ptr<const okvis::kinematics::Transformation> T_SC_0(new okvis::kinematics::Transformation(
      Eigen::Vector3d(0, 0, 0), Eigen::Quaterniond(-sqrt(0.5), 0, 0, sqrt(0.5))));
  std::shared_ptr<const okvis::kinematics::Transformation> T_SC_1(new okvis::kinematics::Transformation(
      Eigen::Vector3d(0.1, 0, 0), Eigen::Quaterniond(-sqrt(0.5), 0, 0, sqrt(0.5))));
  okvis::cameras::NCameraSystem cameraSystem;
  cameraSystem.addCamera(T_SC_0, std::shared_ptr<const okvis::cameras::CameraBase>(Camera::createTestObject()),
                         okvis::cameras::NCameraSystem::DistortionType::Equidistant);
  cameraSystem.addCamera(T_SC_1, std::shared_ptr<const okvis::cameras::CameraBase>(Camera::createTestObject()),
                         okvis::cameras::NCameraSystem::DistortionType::Equidistant);

  okvis::CameraParameters cameraParameters{};
  cameraParameters.online_calibration.do_extrinsics = false;
  okvis::ViSlamBackend backend;
  backend.addCamera(cameraParameters);
  backend.addCamera(cameraParameters);
  backend.addImu(imu);

  // static rig looking at a wall of points 3 m away (camera z = S +y)
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> points;
  std::vector<okvis::LandmarkId> lmIds;
  for (double x = -3.0; x <= 3.0; x += 0.5) {
    for (double z = -3.0; z <= 3.0; z += 0.5) {
      points.push_back(Eigen::Vector4d(x, 3.0, z, 1));
      lmIds.push_back(backend.addLandmark(points.back() + Eigen::Vector4d(0.02, -0.03, 0.01, 0), true));
    }
  }

  std::vector<okvis::StateId> ids;
  std::vector<okvis::StateId> updated;
  const size_t K = 6;
  for (size_t k = 0; k < K; ++k) {
    okvis::MultiFramePtr mf(new okvis::MultiFrame);
    mf->setTimestamp(t0 + okvis::Duration(0.5 * double(k)));
    mf->resetCameraSystemAndFrames(cameraSystem);
    const bool keyframe = (k + 1 < K);  // the last frame is the non-keyframe under test
    ASSERT_TRUE(backend.addStates(mf, imuMeasurements, keyframe));
    const okvis::StateId id(mf->id());
    ids.push_back(id);
    for (size_t i = 0; i < mf->numFrames(); ++i) {
      std::vector<cv::KeyPoint> keypoints;
      std::vector<size_t> lmIdx;
      for (size_t j = 0; j < points.size(); ++j) {
        Eigen::Vector2d px;
        const Eigen::Vector4d p_C = mf->T_SC(i)->inverse() * points[j];  // T_WS = identity
        if (mf->geometryAs<Camera>(i)->projectHomogeneous(p_C, &px)
            == okvis::cameras::ProjectionStatus::Successful) {
          px += Eigen::Vector2d::Random() * 0.5;
          keypoints.push_back(cv::KeyPoint(float(px[0]), float(px[1]), 8.0));
          lmIdx.push_back(j);
        }
      }
      mf->resetKeypoints(i, keypoints);
      for (size_t n = 0; n < lmIdx.size(); ++n) {
        backend.addObservation<Camera>(lmIds[lmIdx[n]], id, i, n);
      }
    }
    if (k + 1 < K) backend.optimiseRealtimeGraph(3, updated, 1, false, false, true);
  }
  ASSERT_FALSE(backend.isKeyframe(ids.back()));
  // perturb the newest state so the cheap solve has something to correct
  const okvis::kinematics::TransformationCacheless T_WS_old = backend.pose(ids.back());
  backend.setPose(ids.back(), okvis::kinematics::TransformationCacheless(
                                  T_WS_old.r() + Eigen::Vector3d(0.03, -0.02, 0.01), T_WS_old.q()));

  const Snapshot before = snapshot(backend, ids);
  updated.clear();
  backend.optimiseRealtimeGraph(3, updated, 1, false, /*onlyNewestState*/ true, true);
  const Snapshot cheap = snapshot(backend, ids);

  ASSERT_EQ(before.landmarks.size(), cheap.landmarks.size());
  for (const auto& lm : before.landmarks) {
    EXPECT_EQ(lm.second, cheap.landmarks.at(lm.first)) << "landmark " << lm.first << " moved";
  }
  for (size_t k = 0; k + 1 < ids.size(); ++k) {
    EXPECT_EQ(before.states.at(ids[k].value()), cheap.states.at(ids[k].value()))
        << "older state " << ids[k].value() << " moved";
  }
  EXPECT_NE(before.states.at(ids.back().value()), cheap.states.at(ids.back().value()));
  EXPECT_LT((backend.pose(ids.back()).r()).norm(), 0.02);  // pulled back toward the truth (origin)
  ASSERT_EQ(updated.size(), 1u);
  EXPECT_EQ(updated.front(), ids.back());

  // realtime_cheap_variable_states = 3: the three newest states may move, older ones and the
  // landmarks stay bit-identical, and all three are reported
  const Snapshot beforeWindow = snapshot(backend, ids);
  updated.clear();
  backend.optimiseRealtimeGraph(3, updated, 1, false, true, true, false, true, 3);
  const Snapshot window = snapshot(backend, ids);
  for (const auto& lm : beforeWindow.landmarks) {
    EXPECT_EQ(lm.second, window.landmarks.at(lm.first)) << "landmark " << lm.first << " moved";
  }
  for (size_t k = 0; k + 3 < ids.size(); ++k) {
    EXPECT_EQ(beforeWindow.states.at(ids[k].value()), window.states.at(ids[k].value()))
        << "state " << ids[k].value() << " outside the window moved";
  }
  EXPECT_EQ(updated.size(), 3u);

  // the keyframe path (full window solve) does move landmarks
  backend.setKeyframe(ids.back(), true);
  backend.optimiseRealtimeGraph(3, updated, 1, false, false, true);
  const Snapshot full = snapshot(backend, ids);
  size_t moved = 0;
  for (const auto& lm : cheap.landmarks) moved += (lm.second != full.landmarks.at(lm.first)) ? 1 : 0;
  EXPECT_GT(moved, 0u);
}
