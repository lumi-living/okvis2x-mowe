/**
 * mow-e (T-0125, ADR-0042 design item 3): WheelOdometryError tests.
 *  1. Model consistency: truth state + noise-free IMU + truth measurement -> residual ~ 0.
 *  2. Analytic vs central-difference Jacobians (as TestGpsError / TestGpsMerge).
 *  3. Slip gates: yaw-rate gate skips, speed gate inflates sigma_v.
 *  4. eliminateStateByImuMerge re-anchors the factor to the previous state (count kept,
 *     residual block kept, same residual).
 *  5. T-0145 wheel scale: Jacobian w.r.t. s, convergence of s on a 5 % over-reading encoder
 *     with the motion known (GNSS) and prior-held without, segment freeze / hand-on in ViGraph.
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
#include <okvis/ceres/WheelOdometryError.hpp>
#include <okvis/kinematics/operators.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/Parameters.hpp>
#include <ceres/manifold.h>
#include <ceres/problem.h>
#include <ceres/solver.h>

namespace {

okvis::ImuParameters testImuParameters() {
  okvis::ImuParameters p;
  p.a0.setZero();
  p.g0.setZero();   // no defaults in ImuParameters (see TestGpsMerge)
  p.s_a.setOnes();
  p.g = 9.81;
  p.a_max = 1000.0;
  p.g_max = 1000.0;
  p.sigma_g_c = 6.0e-4;
  p.sigma_a_c = 2.0e-3;
  p.sigma_gw_c = 3.0e-6;
  p.sigma_aw_c = 2.0e-5;
  p.sigma_bg = 0.01;
  p.sigma_ba = 0.01;
  p.use = true;
  return p;
}

okvis::WheelParameters testWheelParameters() {
  okvis::WheelParameters w;
  // B rotated 30 deg about z and offset from S: exercises C_BS and the omega x r_SB term
  w.T_SB = okvis::kinematics::Transformation(
      Eigen::Vector3d(0.15, -0.05, -0.20),
      Eigen::Quaterniond(Eigen::AngleAxisd(30.0 * M_PI / 180.0, Eigen::Vector3d::UnitZ())));
  w.sigma_v = 0.05;
  w.sigma_lat = 0.5;
  w.sigma_vert = 0.5;
  w.sigma_omega = 0.05;
  w.b_eff = 0.5;
  w.slip_gate_omega = 0.2;
  w.slip_gate_v = 0.3;
  w.loss = "cauchy";
  return w;
}

/// Constant body rate + constant world acceleration, noise-free IMU at 1 kHz.
struct Motion {
  Eigen::Vector3d omega_S{0.2, -0.1, 0.3};
  Eigen::Vector3d a_W{0.3, -0.2, 0.1};
  Eigen::Vector3d v0_W{0.8, 0.1, -0.05};
  okvis::ImuParameters imu = testImuParameters();
  okvis::ImuMeasurementDeque imuMeasurements;
  std::vector<okvis::kinematics::Transformation> T_WS;  // per IMU sample
  double dt = 1e-3;

  Motion(double duration = 0.5) {
    okvis::kinematics::Transformation T(Eigen::Vector3d(1, 2, 3),
                                        Eigen::Quaterniond(Eigen::AngleAxisd(0.7, Eigen::Vector3d(1, 1, 0).normalized())));
    for (size_t i = 0; i <= size_t(duration / dt); ++i) {
      const double t = double(i) * dt;
      T_WS.push_back(T);
      const Eigen::Vector3d f_S = T.C().transpose() * (a_W + Eigen::Vector3d(0, 0, imu.g));
      imuMeasurements.push_back(okvis::ImuMeasurement(okvis::Time(t), okvis::ImuSensorReadings(omega_S, f_S)));
      // exact integration of the constant-rate rotation and constant acceleration
      const Eigen::Vector3d v = v0_W + a_W * t;
      const Eigen::Vector3d r = T.r() + v * dt + 0.5 * a_W * dt * dt;
      const Eigen::Quaterniond q = T.q() * okvis::kinematics::deltaQ(omega_S * dt);
      T = okvis::kinematics::Transformation(r, q);
    }
  }
  Eigen::Vector3d v_W(double t) const { return v0_W + a_W * t; }
  okvis::kinematics::Transformation pose(double t) const { return T_WS.at(size_t(std::lround(t / dt))); }
  /// Truth measurement (v_B,x, omega_B,z) at t.
  Eigen::Vector2d measurement(double t, const okvis::WheelParameters& w) const {
    const Eigen::Matrix3d C_BS = w.T_SB.C().transpose();
    const Eigen::Vector3d v_S = pose(t).C().transpose() * v_W(t);
    const Eigen::Vector3d v_B = C_BS * (v_S + omega_S.cross(w.T_SB.r()));
    return Eigen::Vector2d(v_B[0], (C_BS * omega_S)[2]);
  }
};

}  // namespace

TEST(okvisTestSuite, WheelOdometryErrorModel) {
  Motion m;
  const okvis::WheelParameters w = testWheelParameters();
  const double tk = 0.2, tw = 0.29;
  okvis::ceres::WheelOdometryError term(m.measurement(tw, w),
                                        okvis::ceres::WheelOdometryError::sigmas_t(1, 1, 1, 1),
                                        m.imuMeasurements, m.imu, okvis::Time(tk), okvis::Time(tw), w);
  okvis::ceres::PoseParameterBlock pose(m.pose(tk), 0, okvis::Time(tk));
  okvis::SpeedAndBias sb = okvis::SpeedAndBias::Zero();
  sb.head<3>() = m.v_W(tk);
  okvis::ceres::SpeedAndBiasParameterBlock sbBlock(sb, 0, okvis::Time(tk));
  double scale = 1.0;
  const double* parameters[3] = {pose.parameters(), sbBlock.parameters(), &scale};
  Eigen::Vector4d r;
  ASSERT_TRUE(term.Evaluate(parameters, r.data(), nullptr));
  // Forward speed and yaw rate match the truth (90 ms of 1 kHz trapezoidal preintegration
  // at 0.37 rad/s: sub-mm/s, sub-mrad/s). The lateral/vertical rows are the planar
  // assumption itself: -v_B,y / -v_B,z of the true (non-planar) motion, not zero.
  const Eigen::Matrix3d C_BS = w.T_SB.C().transpose();
  const Eigen::Vector3d v_B_true = C_BS * (m.pose(tw).C().transpose() * m.v_W(tw) + m.omega_S.cross(w.T_SB.r()));
  EXPECT_NEAR(r[0], 0.0, 2e-3) << r.transpose();
  EXPECT_NEAR(r[1], -v_B_true[1], 2e-3) << r.transpose();
  EXPECT_NEAR(r[2], -v_B_true[2], 2e-3) << r.transpose();
  EXPECT_NEAR(r[3], 0.0, 2e-3) << r.transpose();
  EXPECT_GT(std::fabs(v_B_true[1]) + std::fabs(v_B_true[2]), 0.1);  // the motion IS non-planar
  // the measurement is not trivially zero
  EXPECT_GT(std::fabs(term.measurement()[0]), 0.1);
  EXPECT_GT(std::fabs(term.measurement()[1]), 0.1);
  // and a wrong speed shows up in the forward component only (sigma = 1)
  okvis::ceres::WheelOdometryError off(term.measurement() + Eigen::Vector2d(0.5, 0.0),
                                       okvis::ceres::WheelOdometryError::sigmas_t(1, 1, 1, 1),
                                       m.imuMeasurements, m.imu, okvis::Time(tk), okvis::Time(tw), w);
  Eigen::Vector4d r2;
  ASSERT_TRUE(off.Evaluate(parameters, r2.data(), nullptr));
  EXPECT_NEAR(r2[0], 0.5, 2e-3);
  EXPECT_LT((r2.tail<3>() - r.tail<3>()).norm(), 1e-9);
  // T-0145: an encoder over-reading by 5 % is explained exactly by s = 1.05 (forward row only)
  okvis::ceres::WheelOdometryError over(Eigen::Vector2d(1.05 * term.measurement()[0], term.measurement()[1]),
                                        okvis::ceres::WheelOdometryError::sigmas_t(1, 1, 1, 1),
                                        m.imuMeasurements, m.imu, okvis::Time(tk), okvis::Time(tw), w);
  Eigen::Vector4d r3;
  scale = 1.05;
  ASSERT_TRUE(over.Evaluate(parameters, r3.data(), nullptr));
  EXPECT_NEAR(r3[0], 0.0, 3e-3);
  EXPECT_LT((r3.tail<3>() - r.tail<3>()).norm(), 1e-9);
  EXPECT_NEAR(over.speedBx(), term.measurement()[0], 2e-3);
}

TEST(okvisTestSuite, WheelOdometryErrorJacobians) {
  Motion m;
  const okvis::WheelParameters w = testWheelParameters();
  const double tk = 0.2, tw = 0.31;
  okvis::ceres::WheelOdometryError term(m.measurement(tw, w) + Eigen::Vector2d(0.03, -0.02),
                                        okvis::ceres::WheelOdometryError::sigmas_t(0.05, 0.5, 0.5, 0.05),
                                        m.imuMeasurements, m.imu, okvis::Time(tk), okvis::Time(tw), w);
  // disturbed state with non-zero biases
  okvis::kinematics::Transformation T_dist;
  T_dist.setRandom(0.2, 0.1);
  const okvis::kinematics::Transformation T_k = m.pose(tk) * T_dist;
  okvis::SpeedAndBias sb = okvis::SpeedAndBias::Zero();
  sb.head<3>() = m.v_W(tk) + Eigen::Vector3d(0.05, -0.02, 0.03);
  sb.segment<3>(3) = Eigen::Vector3d(0.01, -0.02, 0.015);
  sb.tail<3>() = Eigen::Vector3d(-0.05, 0.02, 0.04);
  okvis::ceres::PoseParameterBlock pose(T_k, 0, okvis::Time(tk));
  okvis::ceres::SpeedAndBiasParameterBlock sbBlock(sb, 0, okvis::Time(tk));
  double scale = 1.04;  // T-0145: away from 1 so the scaled forward row is exercised
  double* parameters[3] = {pose.parameters(), sbBlock.parameters(), &scale};

  Eigen::Matrix<double, 4, 7, Eigen::RowMajor> J0;
  Eigen::Matrix<double, 4, 9, Eigen::RowMajor> J1;
  Eigen::Vector4d J2, J2min;
  Eigen::Matrix<double, 4, 6, Eigen::RowMajor> J0min;
  Eigen::Matrix<double, 4, 9, Eigen::RowMajor> J1min;
  double* jacobians[3] = {J0.data(), J1.data(), J2.data()};
  double* jacobiansMinimal[3] = {J0min.data(), J1min.data(), J2min.data()};
  Eigen::Vector4d residuals;
  // twice: the first call fixes the bias linearisation point
  ASSERT_TRUE(term.EvaluateWithMinimalJacobians(parameters, residuals.data(), jacobians, jacobiansMinimal));
  ASSERT_TRUE(term.EvaluateWithMinimalJacobians(parameters, residuals.data(), jacobians, jacobiansMinimal));

  const double dx = 1e-6;
  okvis::ceres::PoseManifold poseManifold;
  Eigen::Matrix<double, 4, 6> J0n;
  for (int i = 0; i < 6; ++i) {
    Eigen::Matrix<double, 6, 1> d = Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Vector4d rp, rm;
    d[i] = dx;  poseManifold.Plus(parameters[0], d.data(), parameters[0]); term.Evaluate(parameters, rp.data(), nullptr); pose.setEstimate(T_k);
    d[i] = -dx; poseManifold.Plus(parameters[0], d.data(), parameters[0]); term.Evaluate(parameters, rm.data(), nullptr); pose.setEstimate(T_k);
    J0n.col(i) = (rp - rm) / (2.0 * dx);
  }
  Eigen::Matrix<double, 4, 9> J1n;
  for (int i = 0; i < 9; ++i) {
    Eigen::Matrix<double, 9, 1> d = Eigen::Matrix<double, 9, 1>::Zero();
    Eigen::Vector4d rp, rm;
    d[i] = dx;  sbBlock.plus(parameters[1], d.data(), parameters[1]); term.Evaluate(parameters, rp.data(), nullptr); sbBlock.setEstimate(sb);
    d[i] = -dx; sbBlock.plus(parameters[1], d.data(), parameters[1]); term.Evaluate(parameters, rm.data(), nullptr); sbBlock.setEstimate(sb);
    J1n.col(i) = (rp - rm) / (2.0 * dx);
  }
  Eigen::Vector4d J2n;
  {
    Eigen::Vector4d rp, rm;
    scale = 1.04 + dx; term.Evaluate(parameters, rp.data(), nullptr);
    scale = 1.04 - dx; term.Evaluate(parameters, rm.data(), nullptr);
    scale = 1.04;
    J2n = (rp - rm) / (2.0 * dx);
  }
  EXPECT_LT((J2n - J2).norm() / J2.norm(), 1e-6) << "J2 " << J2.transpose() << " num " << J2n.transpose();
  EXPECT_TRUE(J2min.isApprox(J2));
  EXPECT_LT(J2.tail<3>().norm(), 1e-15);
  Eigen::Matrix<double, 6, 7, Eigen::RowMajor> lift;
  okvis::ceres::PoseManifold::minusJacobian(parameters[0], lift.data());
  // whitened by 1/0.05 = 20; relative bounds as in TestGpsMerge
  EXPECT_LT((J0n - J0min).norm() / J0min.norm(), 1e-4) << "J0min\n" << J0min << "\nnum\n" << J0n;
  EXPECT_LT((J0n * lift - J0).norm() / J0.norm(), 1e-4) << "J0\n" << J0 << "\nnum\n" << J0n * lift;
  EXPECT_LT((J1n - J1).norm() / J1.norm(), 1e-4) << "J1\n" << J1 << "\nnum\n" << J1n;
  EXPECT_TRUE(J1min.isApprox(J1));
  // the pose-position columns are exactly zero: velocity does not depend on where we are
  EXPECT_LT(J0min.leftCols<3>().norm(), 1e-12);
}

namespace {

/// Three-state graph on a static, level IMU (as TestGpsMerge).
struct GraphFixture {
  okvis::ImuParameters imuParameters = testImuParameters();
  okvis::WheelParameters wheelParameters = testWheelParameters();
  okvis::ImuMeasurementDeque imuMeasurements;
  std::shared_ptr<okvis::cameras::NCameraSystem> cameraSystem;
  okvis::Time t0{1000.0};
  okvis::StateId ids[3];

  GraphFixture(okvis::ViGraphEstimator& graph) {
    const double imuRate = 200.0;
    for (size_t i = 0; i <= size_t(1.5 * imuRate); ++i) {
      imuMeasurements.push_back(okvis::ImuMeasurement(
          t0 + okvis::Duration(double(i) / imuRate),
          okvis::ImuSensorReadings(Eigen::Vector3d::Zero(), Eigen::Vector3d(0, 0, imuParameters.g))));
    }
    std::shared_ptr<const okvis::kinematics::Transformation> T_SC(
        new okvis::kinematics::Transformation(Eigen::Vector3d(0, 0, 0),
                                              Eigen::Quaterniond(-sqrt(0.5), 0, 0, sqrt(0.5))));
    std::shared_ptr<const okvis::cameras::CameraBase> geometry(
        okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion>::createTestObject());
    cameraSystem.reset(new okvis::cameras::NCameraSystem);
    cameraSystem->addCamera(T_SC, geometry, okvis::cameras::NCameraSystem::DistortionType::Equidistant);
    okvis::CameraParameters cameraParameters;
    graph.addCamera(cameraParameters);
    graph.addImu(imuParameters);
    graph.addWheel(wheelParameters);
    ids[0] = graph.addStatesInitialise(t0, imuMeasurements, *cameraSystem);
    ids[1] = graph.addStatesPropagate(t0 + okvis::Duration(0.4), imuMeasurements, false);
    ids[2] = graph.addStatesPropagate(t0 + okvis::Duration(0.8), imuMeasurements, true);
  }

  okvis::WheelMeasurement measurementAfter(okvis::ViGraphEstimator& graph, okvis::StateId id,
                                           double vLeft, double vRight, int slip = 0) {
    okvis::WheelMeasurement m;
    m.timeStamp = graph.timestamp(id) + okvis::Duration(0.1);
    m.measurement = okvis::WheelSensorReadings(vLeft, vRight, wheelParameters.b_eff, slip);
    return m;
  }

  Eigen::Vector4d residual(const okvis::ViGraphEstimator& graph, okvis::StateId id,
                           const okvis::ceres::WheelOdometryError& term) {
    okvis::ceres::PoseParameterBlock pose(graph.pose(id), 0, graph.timestamp(id));
    okvis::ceres::SpeedAndBiasParameterBlock sb(graph.speedAndBias(id), 0, graph.timestamp(id));
    const double scale = 1.0;
    const double* parameters[3] = {pose.parameters(), sb.parameters(), &scale};
    Eigen::Vector4d r;
    EXPECT_TRUE(term.Evaluate(parameters, r.data(), nullptr));
    return r;
  }
};

}  // namespace

TEST(okvisTestSuite, WheelOdometrySlipGates) {
  okvis::ViGraphEstimator graph;
  GraphFixture f(graph);
  // consistent with the static graph: v = 0, omega = 0 -> plain factor
  ASSERT_TRUE(graph.addWheelMeasurement(f.ids[1], f.measurementAfter(graph, f.ids[1], 0.0, 0.0), f.imuMeasurements));
  EXPECT_EQ(graph.wheelFactorStats().added, 1u);
  EXPECT_DOUBLE_EQ(graph.wheelErrorTerm(f.ids[1], 0)->sigmas()[0], f.wheelParameters.sigma_v);
  // encoder yaw rate 2 rad/s against a still gyro -> skipped by the yaw-rate gate
  ASSERT_FALSE(graph.addWheelMeasurement(f.ids[1], f.measurementAfter(graph, f.ids[1], -0.5, 0.5), f.imuMeasurements));
  EXPECT_EQ(graph.wheelFactorStats().gatedOmega, 1u);
  EXPECT_EQ(graph.numWheelFactors(f.ids[1]), 1u);
  // encoder speed 1 m/s against a still state -> added with sigma_v x 4
  ASSERT_TRUE(graph.addWheelMeasurement(f.ids[1], f.measurementAfter(graph, f.ids[1], 1.0, 1.0), f.imuMeasurements));
  EXPECT_EQ(graph.wheelFactorStats().gatedV, 1u);
  EXPECT_EQ(graph.numWheelFactors(f.ids[1]), 2u);
  EXPECT_DOUBLE_EQ(graph.wheelErrorTerm(f.ids[1], 1)->sigmas()[0], 4.0 * f.wheelParameters.sigma_v);
  EXPECT_EQ(graph.wheelFactorStats().gatedTimesNs.size(), 2u);
  // publisher slip flag inflates v and omega sigmas
  ASSERT_TRUE(graph.addWheelMeasurement(f.ids[1], f.measurementAfter(graph, f.ids[1], 0.0, 0.0, 1), f.imuMeasurements));
  EXPECT_EQ(graph.wheelFactorStats().slipFlagged, 1u);
  EXPECT_DOUBLE_EQ(graph.wheelErrorTerm(f.ids[1], 2)->sigmas()[3], 3.0 * f.wheelParameters.sigma_omega);
  // measurement before the IMU coverage / no state -> the deque walk skips it
  okvis::WheelMeasurementDeque deque;
  deque.push_back(f.measurementAfter(graph, f.ids[2], 0.0, 0.0));  // after the last state: attaches to ids[2]
  std::deque<okvis::StateId> sids;
  ASSERT_TRUE(graph.addWheelMeasurements(deque, f.imuMeasurements, &sids));
  ASSERT_EQ(sids.size(), 1u);
  EXPECT_EQ(sids[0], f.ids[2]);
  EXPECT_EQ(graph.numWheelFactors(), 4u);
}

TEST(okvisTestSuite, WheelMergeKeepsFactorOnPreviousState) {
  okvis::ViGraphEstimator graph;
  GraphFixture f(graph);
  const okvis::WheelMeasurement m = f.measurementAfter(graph, f.ids[1], 0.02, -0.02);  // v 0, omega -0.08
  ASSERT_TRUE(graph.addWheelMeasurement(f.ids[1], m, f.imuMeasurements));
  size_t inProblem = 0;
  ASSERT_EQ(graph.numWheelFactors(f.ids[1], &inProblem), 1u);
  ASSERT_EQ(inProblem, 1u);
  auto oldTerm = graph.wheelErrorTerm(f.ids[1], 0);
  ASSERT_TRUE(oldTerm);
  const Eigen::Vector4d r_old = f.residual(graph, f.ids[1], *oldTerm);

  ASSERT_TRUE(graph.removeAllObservations(f.ids[1]));
  ASSERT_TRUE(graph.eliminateStateByImuMerge(f.ids[1], f.ids[2]));

  EXPECT_EQ(graph.numWheelFactors(f.ids[1]), 0u);
  inProblem = 0;
  EXPECT_EQ(graph.numWheelFactors(f.ids[0], &inProblem), 1u);
  EXPECT_EQ(inProblem, 1u);
  EXPECT_EQ(graph.numWheelFactors(), 1u);
  EXPECT_EQ(graph.wheelFactorStats().merged, 1u);
  EXPECT_EQ(graph.wheelFactorStats().dropped, 0u);
  auto newTerm = graph.wheelErrorTerm(f.ids[0], 0);
  ASSERT_TRUE(newTerm);
  EXPECT_EQ(newTerm->tk(), graph.timestamp(f.ids[0]));
  EXPECT_EQ(newTerm->tw(), m.timeStamp);
  EXPECT_TRUE(newTerm->measurement().isApprox(oldTerm->measurement()));
  EXPECT_TRUE(newTerm->sigmas().isApprox(oldTerm->sigmas()));
  const Eigen::Vector4d r_new = f.residual(graph, f.ids[0], *newTerm);
  EXPECT_LT((r_new - r_old).norm(), 1e-2) << "old " << r_old.transpose() << " new " << r_new.transpose();
  // the yaw-rate residual is the -0.08 rad/s encoder yaw rate whitened by 1/0.05
  EXPECT_NEAR(r_new[3], -0.08 / f.wheelParameters.sigma_omega, 1e-6);
}

namespace {

/// T-0145: a 20 s drive (the Motion trajectory sampled every 0.1 s, one wheel factor per state
/// 0.05 s after it) with encoders over-reading by 5 %; scale prior N(1, 0.05) as in the configs.
struct ScaleProblem {
  Motion m{20.0};
  okvis::WheelParameters w = testWheelParameters();
  std::vector<std::unique_ptr<okvis::ceres::PoseParameterBlock>> poses;
  std::vector<std::unique_ptr<okvis::ceres::SpeedAndBiasParameterBlock>> speeds;
  std::vector<std::unique_ptr<okvis::ceres::WheelOdometryError>> terms;
  okvis::ceres::WheelScalePrior prior{1.0, 0.05};
  okvis::ceres::PoseManifold poseManifold;
  ::ceres::SubsetManifold velocityOnly{9, {3, 4, 5, 6, 7, 8}};  // biases constant
  double scale = 1.0;
  ::ceres::Problem::Options options() {
    ::ceres::Problem::Options o;
    o.cost_function_ownership = ::ceres::DO_NOT_TAKE_OWNERSHIP;
    o.manifold_ownership = ::ceres::DO_NOT_TAKE_OWNERSHIP;
    return o;
  }
  ::ceres::Problem problem{options()};

  /// \param motionKnown Poses and velocities constant at the truth (what RTK-GNSS pins down);
  ///        otherwise the velocities are free and only the wheel factors and the prior remain.
  explicit ScaleProblem(bool motionKnown) {
    for (double tk = 0.1; tk < 19.8; tk += 0.1) {
      const double tw = tk + 0.05;
      okvis::SpeedAndBias sb = okvis::SpeedAndBias::Zero();
      sb.head<3>() = m.v_W(tk);
      poses.emplace_back(new okvis::ceres::PoseParameterBlock(m.pose(tk), 0, okvis::Time(tk)));
      speeds.emplace_back(new okvis::ceres::SpeedAndBiasParameterBlock(sb, 0, okvis::Time(tk)));
      Eigen::Vector2d z = m.measurement(tw, w);
      z[0] *= 1.05;
      terms.emplace_back(new okvis::ceres::WheelOdometryError(
          z, okvis::ceres::WheelOdometryError::sigmas_t(w.sigma_v, w.sigma_lat, w.sigma_vert, w.sigma_omega),
          m.imuMeasurements, m.imu, okvis::Time(tk), okvis::Time(tw), w));
      problem.AddParameterBlock(poses.back()->parameters(), 7, &poseManifold);
      problem.AddResidualBlock(terms.back().get(), nullptr, poses.back()->parameters(),
                               speeds.back()->parameters(), &scale);
      problem.SetParameterBlockConstant(poses.back()->parameters());
      if (motionKnown) {
        problem.SetParameterBlockConstant(speeds.back()->parameters());
      } else {  // biases stay put, the velocity is free
        problem.SetManifold(speeds.back()->parameters(), &velocityOnly);
      }
    }
    problem.AddResidualBlock(&prior, nullptr, &scale);
  }
  void solve() {
    ::ceres::Solver::Options o;
    o.max_num_iterations = 50;
    o.linear_solver_type = ::ceres::SPARSE_NORMAL_CHOLESKY;
    ::ceres::Solver::Summary summary;
    ::ceres::Solve(o, &problem, &summary);
    ASSERT_TRUE(summary.IsSolutionUsable()) << summary.BriefReport();
  }
};

}  // namespace

TEST(okvisTestSuite, WheelScaleConvergesWithKnownMotion) {
  ScaleProblem p(true);
  p.solve();
  EXPECT_NEAR(p.scale, 1.05, 0.01);
  EXPECT_NEAR(p.scale, 1.05, 1e-3);  // 200 factors at |v| ~ 1 m/s, sigma_v 0.05: the prior is negligible
}

TEST(okvisTestSuite, WheelScaleHeldByPriorWithoutGnss) {
  ScaleProblem p(false);
  p.solve();
  // the velocities absorb the 5 %; with nothing else observing the speed the prior holds s
  EXPECT_NEAR(p.scale, 1.0, 0.01);
}

TEST(okvisTestSuite, WheelScaleSegmentsFreezeAndHandOn) {
  okvis::ViGraphEstimator graph;
  okvis::ImuParameters imuParameters = testImuParameters();
  okvis::WheelParameters w = testWheelParameters();
  w.scale = 1.02;
  w.scale_sigma = 0.05;
  w.scale_walk = 0.01;
  w.scale_segment_s = 0.3;  // fixture states at t0 + {0, 0.4, 0.8} s, t0 = 1000 s
  graph.addImu(imuParameters);
  okvis::CameraParameters cameraParameters;
  graph.addCamera(cameraParameters);
  graph.addWheel(w);
  // (not GraphFixture: it adds the default wheel parameters)
  okvis::ImuMeasurementDeque imu;
  const okvis::Time t0(1000.0);
  for (size_t i = 0; i <= 300; ++i) {
    imu.push_back(okvis::ImuMeasurement(t0 + okvis::Duration(double(i) / 200.0),
        okvis::ImuSensorReadings(Eigen::Vector3d::Zero(), Eigen::Vector3d(0, 0, imuParameters.g))));
  }
  std::shared_ptr<const okvis::kinematics::Transformation> T_SC(new okvis::kinematics::Transformation());
  std::shared_ptr<const okvis::cameras::CameraBase> geometry(
      okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion>::createTestObject());
  okvis::cameras::NCameraSystem cameras;
  cameras.addCamera(T_SC, geometry, okvis::cameras::NCameraSystem::DistortionType::Equidistant);
  const okvis::StateId id0 = graph.addStatesInitialise(t0, imu, cameras);
  const okvis::StateId id1 = graph.addStatesPropagate(t0 + okvis::Duration(0.4), imu, false);
  const okvis::StateId id2 = graph.addStatesPropagate(t0 + okvis::Duration(0.8), imu, true);
  auto meas = [&](okvis::StateId id, double dt) {
    okvis::WheelMeasurement m;
    m.timeStamp = graph.timestamp(id) + okvis::Duration(dt);
    m.measurement = okvis::WheelSensorReadings(0.0, 0.0, w.b_eff, 0);
    return m;
  };
  EXPECT_DOUBLE_EQ(graph.wheelScale(), 1.02);  // before any measurement: the configured value
  ASSERT_TRUE(graph.addWheelMeasurement(id0, meas(id0, 0.1), imu));    // 1000.1 s: segment 3333
  ASSERT_TRUE(graph.addWheelMeasurement(id1, meas(id1, 0.05), imu));   // 1000.45 s: segment 3334
  auto seg = graph.wheelScaleSegments();
  ASSERT_EQ(seg.size(), 2u);
  EXPECT_TRUE(seg[0].frozen);
  EXPECT_FALSE(seg[1].frozen);
  EXPECT_DOUBLE_EQ(seg[0].sigma, 0.05);
  // a standing robot carries no scale information: the frozen sigma is the prior's
  EXPECT_NEAR(seg[0].frozenSigma, 0.05, 1e-12);
  EXPECT_NEAR(seg[1].sigma, std::sqrt(0.05 * 0.05 + 0.01 * 0.01 * 0.3), 1e-12);
  EXPECT_DOUBLE_EQ(seg[1].value, 1.02);
  EXPECT_NEAR(seg[0].start.toSec(), 3333 * 0.3, 1e-6);
  EXPECT_EQ(graph.variableWheelScaleBlock() != nullptr, true);
  // a late measurement for the frozen segment uses it, does not open a new one
  ASSERT_TRUE(graph.addWheelMeasurement(id0, meas(id0, 0.15), imu));
  EXPECT_EQ(graph.wheelScaleSegments().size(), 2u);
  // the elimination merge re-anchors id1's factor onto id0 in the same (newest) segment
  ASSERT_TRUE(graph.removeAllObservations(id1));
  ASSERT_TRUE(graph.eliminateStateByImuMerge(id1, id2));
  EXPECT_EQ(graph.numWheelFactors(id0), 3u);
  EXPECT_EQ(graph.wheelScaleSegments().size(), 2u);
  // a mirror graph keeps every segment constant and copies the values
  okvis::ViGraphEstimator mirror;
  mirror.addImu(imuParameters);
  mirror.addCamera(cameraParameters);
  mirror.setWheelScaleMirror(true);
  mirror.addWheel(w);
  const okvis::StateId m0 = mirror.addStatesInitialise(t0, imu, cameras);
  ASSERT_TRUE(mirror.addWheelMeasurement(m0, meas(m0, 0.1), imu));
  EXPECT_EQ(mirror.variableWheelScaleBlock(), nullptr);
  EXPECT_TRUE(mirror.wheelScaleSegments()[0].frozen);
  EXPECT_DOUBLE_EQ(mirror.wheelScaleSegments()[0].sigma, 0.0);
  *graph.variableWheelScaleBlock() = 1.07;  // newest realtime segment 3334; the mirror only has 3333
  mirror.setWheelScalesFrom(graph);
  EXPECT_DOUBLE_EQ(mirror.wheelScale(), 1.02);
  ASSERT_TRUE(mirror.addWheelMeasurement(m0, meas(m0, 0.4), imu));   // opens 3334 at the last value
  mirror.setWheelScalesFrom(graph);
  EXPECT_DOUBLE_EQ(mirror.wheelScale(), 1.07);
}
