/**
 * mow-e (T-0117, ADR-0042 design item 2): GNSS factors attached to a non-keyframe
 * must survive ViGraphEstimator::eliminateStateByImuMerge() -- re-anchored to the
 * previous state with the IMU preintegration extended to the GNSS time. Upstream
 * OKVIS2-X 38043e4 discarded them (the carry-forward was commented out).
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */
#include <gtest/gtest.h>
#include <okvis/ViGraphEstimator.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/SpeedAndBiasParameterBlock.hpp>
#include <okvis/ceres/GpsErrorAsynchronous.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/Parameters.hpp>

namespace {

struct Fixture {
  okvis::ImuParameters imuParameters;
  okvis::GpsParameters gpsParameters;
  okvis::ImuMeasurementDeque imuMeasurements;
  std::shared_ptr<okvis::cameras::NCameraSystem> cameraSystem;
  okvis::Time t0;

  Fixture() {
    imuParameters.a0.setZero();
    imuParameters.g = 9.81;
    imuParameters.a_max = 1000.0;
    imuParameters.g_max = 1000.0;
    const double imuRate = 200.0;
    imuParameters.sigma_g_c = 6.0e-4;
    imuParameters.sigma_a_c = 2.0e-3;
    imuParameters.sigma_gw_c = 3.0e-6;
    imuParameters.sigma_aw_c = 2.0e-5;
    imuParameters.sigma_bg = 0.01;
    imuParameters.sigma_ba = 0.01;
    imuParameters.use = true;

    gpsParameters.type = "cartesian";
    gpsParameters.r_SA = Eigen::Vector3d(0.05, -0.02, 0.10);
    gpsParameters.yawErrorThreshold = 1.0;
    gpsParameters.robustGpsInit = false;

    // 1.5 s of a static, level IMU at 200 Hz: specific force = +g along z_S, no rotation.
    t0 = okvis::Time(1000.0);
    const double dt = 1.0 / imuRate;
    for (size_t i = 0; i <= size_t(1.5 * imuRate); ++i) {
      imuMeasurements.push_back(okvis::ImuMeasurement(
          t0 + okvis::Duration(dt * double(i)),
          okvis::ImuSensorReadings(Eigen::Vector3d::Zero(),
                                   Eigen::Vector3d(0, 0, imuParameters.g))));
    }

    std::shared_ptr<const okvis::kinematics::Transformation> T_SC(
        new okvis::kinematics::Transformation(Eigen::Vector3d(0, 0, 0),
                                              Eigen::Quaterniond(-sqrt(0.5), 0, 0, sqrt(0.5))));
    std::shared_ptr<const okvis::cameras::CameraBase> geometry(
        okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion>::createTestObject());
    cameraSystem.reset(new okvis::cameras::NCameraSystem);
    cameraSystem->addCamera(T_SC, geometry,
                            okvis::cameras::NCameraSystem::DistortionType::Equidistant);
  }

  // Three states: t0 (keyframe, first), t0+0.4 (non-keyframe), t0+0.8 (keyframe).
  void addStates(okvis::ViGraphEstimator& graph, okvis::StateId ids[3]) {
    okvis::CameraParameters cameraParameters;
    graph.addCamera(cameraParameters);
    graph.addImu(imuParameters);
    graph.addGps(gpsParameters);
    ids[0] = graph.addStatesInitialise(t0, imuMeasurements, *cameraSystem);
    ids[1] = graph.addStatesPropagate(t0 + okvis::Duration(0.4), imuMeasurements, false);
    ids[2] = graph.addStatesPropagate(t0 + okvis::Duration(0.8), imuMeasurements, true);
  }

  // A fix 0.1 s after the middle state, consistent with the graph's own estimate.
  okvis::GpsMeasurement fixAfter(okvis::ViGraphEstimator& graph, okvis::StateId id) {
    okvis::GpsMeasurement m;
    m.timeStamp = graph.timestamp(id) + okvis::Duration(0.1);
    const okvis::kinematics::Transformation T_WS = graph.pose(id);
    const Eigen::Vector3d p_A_W = T_WS.r() + T_WS.C() * gpsParameters.r_SA;
    const okvis::kinematics::Transformation T_GW = graph.T_GW();
    const Eigen::Vector3d z = T_GW.C() * p_A_W + T_GW.r() + Eigen::Vector3d(0.01, -0.005, 0.02);
    m.measurement = okvis::GpsSensorReadings(z, 0.02, 0.02, 0.04);
    return m;
  }

  // Whitened residual of a term at the graph's current estimates of a state.
  Eigen::Vector3d residual(const okvis::ViGraphEstimator& graph, okvis::StateId id,
                           const okvis::ceres::GpsErrorAsynchronous& term) {
    okvis::ceres::PoseParameterBlock pose(graph.pose(id), 0, graph.timestamp(id));
    okvis::ceres::SpeedAndBiasParameterBlock sb(graph.speedAndBias(id), 0, graph.timestamp(id));
    okvis::ceres::PoseParameterBlock T_GW(graph.T_GW(), 0, graph.timestamp(id));
    const double* parameters[3] = {pose.parameters(), sb.parameters(), T_GW.parameters()};
    Eigen::Vector3d r;
    EXPECT_TRUE(term.Evaluate(parameters, r.data(), nullptr));
    return r;
  }
};

}  // namespace

TEST(okvisTestSuite, GpsMergeKeepsFactorOnPreviousState) {
  Fixture f;
  okvis::ViGraphEstimator graph;
  okvis::StateId ids[3];
  f.addStates(graph, ids);
  graph.setGpsStatus(okvis::gpsStatus::Initialised);  // factors go straight into the problem

  okvis::GpsMeasurement m = f.fixAfter(graph, ids[1]);
  ASSERT_TRUE(graph.addGpsMeasurement(ids[1], m, f.imuMeasurements));
  size_t inProblem = 0;
  ASSERT_EQ(graph.numGpsFactors(ids[1], &inProblem), 1u);
  ASSERT_EQ(inProblem, 1u);
  ASSERT_EQ(graph.gpsFactorStats().added, 1u);
  auto oldTerm = graph.gpsErrorTerm(ids[1], 0);
  ASSERT_TRUE(oldTerm);
  const Eigen::Vector3d r_old = f.residual(graph, ids[1], *oldTerm);

  // eliminate the middle (non-keyframe) state exactly as ViSlamBackend::eliminateImuFrames does
  ASSERT_TRUE(graph.removeAllObservations(ids[1]));
  ASSERT_TRUE(graph.eliminateStateByImuMerge(ids[1], ids[2]));

  EXPECT_EQ(graph.numGpsFactors(ids[1]), 0u);  // state is gone
  inProblem = 0;
  EXPECT_EQ(graph.numGpsFactors(ids[0], &inProblem), 1u);  // factor now on the previous state
  EXPECT_EQ(inProblem, 1u);                                // ... and still in the ceres problem
  EXPECT_EQ(graph.numGpsFactors(), 1u);
  EXPECT_EQ(graph.gpsFactorStats().merged, 1u);
  EXPECT_EQ(graph.gpsFactorStats().dropped, 0u);

  okvis::AlignedVector<Eigen::Vector3d> measurements;
  graph.gpsMeasurements(ids[0], measurements);
  ASSERT_EQ(measurements.size(), 1u);
  EXPECT_TRUE(measurements[0].isApprox(m.measurement.position));

  auto newTerm = graph.gpsErrorTerm(ids[0], 0);
  ASSERT_TRUE(newTerm);
  EXPECT_EQ(newTerm->tk(), graph.timestamp(ids[0]));  // re-anchored ...
  EXPECT_EQ(newTerm->tg(), m.timeStamp);              // ... to the same GNSS time
  EXPECT_TRUE(newTerm->information().isApprox(oldTerm->information()));

  // Same residual from the new anchor: the previous state's estimate propagated over the
  // merged IMU deque lands where the eliminated state's estimate did (same IMU).
  const Eigen::Vector3d r_new = f.residual(graph, ids[0], *newTerm);
  EXPECT_LT((r_new - r_old).norm(), 1e-2) << "old " << r_old.transpose() << " new " << r_new.transpose();

  // Analytic vs numeric Jacobians of the merged factor (as in TestGpsError).
  okvis::ceres::PoseParameterBlock pose(graph.pose(ids[0]), 0, graph.timestamp(ids[0]));
  okvis::ceres::SpeedAndBiasParameterBlock sb(graph.speedAndBias(ids[0]), 0, graph.timestamp(ids[0]));
  okvis::ceres::PoseParameterBlock T_GW(graph.T_GW(), 0, graph.timestamp(ids[0]));
  double* parameters[3] = {pose.parameters(), sb.parameters(), T_GW.parameters()};
  Eigen::Matrix<double, 3, 7, Eigen::RowMajor> J0, J2;
  Eigen::Matrix<double, 3, 9, Eigen::RowMajor> J1;
  double* jacobians[3] = {J0.data(), J1.data(), J2.data()};
  Eigen::Vector3d residuals;
  ASSERT_TRUE(newTerm->EvaluateWithMinimalJacobians(parameters, residuals.data(), jacobians, nullptr));
  ASSERT_TRUE(newTerm->EvaluateWithMinimalJacobians(parameters, residuals.data(), jacobians, nullptr));
  const double dx = 1e-6;
  okvis::ceres::PoseManifold poseManifold;
  auto numDiffPose = [&](double* block, const okvis::kinematics::Transformation& reset,
                         okvis::ceres::PoseParameterBlock& pb, Eigen::Matrix<double, 3, 6>& J) {
    for (int i = 0; i < 6; ++i) {
      Eigen::Matrix<double, 6, 1> d = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Vector3d rp, rm;
      d[i] = dx;  poseManifold.Plus(block, d.data(), block); newTerm->Evaluate(parameters, rp.data(), nullptr); pb.setEstimate(reset);
      d[i] = -dx; poseManifold.Plus(block, d.data(), block); newTerm->Evaluate(parameters, rm.data(), nullptr); pb.setEstimate(reset);
      J.col(i) = (rp - rm) / (2.0 * dx);
    }
  };
  Eigen::Matrix<double, 3, 6> J0n, J2n;
  numDiffPose(parameters[0], graph.pose(ids[0]), pose, J0n);
  numDiffPose(parameters[2], graph.T_GW(), T_GW, J2n);
  Eigen::Matrix<double, 6, 7, Eigen::RowMajor> lift0, lift2;
  okvis::ceres::PoseManifold::minusJacobian(parameters[0], lift0.data());
  okvis::ceres::PoseManifold::minusJacobian(parameters[2], lift2.data());
  // Residuals are whitened by 1/sigma = 50, so compare relative to the Jacobian norm
  // (~87 here); 1e-4 relative = 0.009 absolute, central differences with dx = 1e-6.
  EXPECT_LT((J0n * lift0 - J0).norm() / J0.norm(), 1e-4) << (J0n * lift0 - J0).norm();
  EXPECT_LT((J2n * lift2 - J2).norm() / J2.norm(), 1e-4) << (J2n * lift2 - J2).norm();
  Eigen::Matrix<double, 3, 9> J1n;
  const okvis::SpeedAndBias sb0 = graph.speedAndBias(ids[0]);
  for (int i = 0; i < 9; ++i) {
    Eigen::Matrix<double, 9, 1> d = Eigen::Matrix<double, 9, 1>::Zero();
    Eigen::Vector3d rp, rm;
    d[i] = dx;  sb.plus(parameters[1], d.data(), parameters[1]); newTerm->Evaluate(parameters, rp.data(), nullptr); sb.setEstimate(sb0);
    d[i] = -dx; sb.plus(parameters[1], d.data(), parameters[1]); newTerm->Evaluate(parameters, rm.data(), nullptr); sb.setEstimate(sb0);
    J1n.col(i) = (rp - rm) / (2.0 * dx);
  }
  EXPECT_LT((J1n - J1).norm() / J1.norm(), 1e-4) << (J1n - J1).norm();
}

TEST(okvisTestSuite, GpsMergeKeepsBufferedInitFactor) {
  // Before T_GW is observable (Idle) the factor is stored but not yet a residual block;
  // addGpsInitFactors() later adds the blocks of every state in gpsInitMap_. The merge
  // must re-key that map, or the buffered fix is never optimised.
  Fixture f;
  okvis::ViGraphEstimator graph;
  okvis::StateId ids[3];
  f.addStates(graph, ids);
  graph.setGpsStatus(okvis::gpsStatus::Idle);
  okvis::GpsMeasurementDeque fixes;
  fixes.push_back(f.fixAfter(graph, ids[1]));
  ASSERT_TRUE(graph.addGpsMeasurements(fixes, f.imuMeasurements, nullptr));  // Idle path -> gpsInitMap_
  size_t inProblem = 1;
  ASSERT_EQ(graph.numGpsFactors(ids[1], &inProblem), 1u);
  ASSERT_EQ(inProblem, 0u);

  ASSERT_TRUE(graph.removeAllObservations(ids[1]));
  ASSERT_TRUE(graph.eliminateStateByImuMerge(ids[1], ids[2]));
  inProblem = 1;
  EXPECT_EQ(graph.numGpsFactors(ids[0], &inProblem), 1u);
  EXPECT_EQ(inProblem, 0u);  // still buffered, not in the problem

  graph.setGpsStatus(okvis::gpsStatus::Initialising);
  graph.addGpsInitFactors();
  inProblem = 0;
  EXPECT_EQ(graph.numGpsFactors(ids[0], &inProblem), 1u);
  EXPECT_EQ(inProblem, 1u);  // re-keyed gpsInitMap_ entry got its residual block
}
