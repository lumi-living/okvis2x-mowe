/**
 * mow-e (T-0125, ADR-0042 design item 3): WheelOdometryError implementation.
 * Preintegration is the GpsErrorAsynchronous::redoPreintegration scheme without the
 * position double integrals and without the covariance propagation.
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

#include <okvis/kinematics/operators.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>
#include <okvis/ceres/WheelOdometryError.hpp>
#include <glog/logging.h>

namespace okvis {
namespace ceres {

WheelOdometryError::WheelOdometryError(const measurement_t& measurement, const sigmas_t& sigmas,
                                       const okvis::ImuMeasurementDeque& imuMeasurements,
                                       const okvis::ImuParameters& imuParameters,
                                       const okvis::Time& tk, const okvis::Time& tw,
                                       const okvis::WheelParameters& wheelParameters)
    : measurement_(measurement), sigmas_(sigmas), wheelParameters_(wheelParameters),
      imuParameters_(imuParameters), imuMeasurements_(imuMeasurements), tk_(tk), tw_(tw) {
  sqrtInformationDiag_ = sigmas_.cwiseInverse();
  omega_S_tw_raw_ = gyroAt(imuMeasurements_, tw_);
}

Eigen::Vector3d WheelOdometryError::gyroAt(const okvis::ImuMeasurementDeque& imu,
                                           const okvis::Time& t) {
  if (imu.empty()) return Eigen::Vector3d::Zero();
  if (t <= imu.front().timeStamp) return imu.front().measurement.gyroscopes;
  if (t >= imu.back().timeStamp) return imu.back().measurement.gyroscopes;
  auto it = std::lower_bound(imu.begin(), imu.end(), t,
                             [](const ImuMeasurement& m, const okvis::Time& tt) { return m.timeStamp < tt; });
  const auto prev = std::prev(it);
  const double span = (it->timeStamp - prev->timeStamp).toSec();
  const double r = span > 0.0 ? (t - prev->timeStamp).toSec() / span : 0.0;
  return (1.0 - r) * prev->measurement.gyroscopes + r * it->measurement.gyroscopes;
}

bool WheelOdometryError::Evaluate(double const* const* parameters, double* residuals,
                                  double** jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, nullptr);
}

bool WheelOdometryError::EvaluateWithMinimalJacobians(double const* const* parameters,
                                                      double* residuals, double** jacobians,
                                                      double** jacobiansMinimal) const {
  // state at tk
  Eigen::Map<const Eigen::Vector3d> r_WS(&parameters[0][0]);
  const Eigen::Quaterniond q_WS(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
  okvis::SpeedAndBias speedAndBiases;
  for (size_t i = 0; i < 9; ++i) speedAndBiases[i] = parameters[1][i];
  const Eigen::Matrix3d C_WS_tk = q_WS.toRotationMatrix();

  // ----- preintegration tk -> tw (same redo policy as GpsErrorAsynchronous) -----
  const double Delta_t = (tw_ - tk_).toSec();
  Eigen::Matrix<double, 6, 1> Delta_b;
  std::lock_guard<std::mutex> lock(preintegrationMutex_);
  Delta_b = speedAndBiases.tail<6>() - speedAndBiases_ref_.tail<6>();
  redo_ = redo_ || (Delta_b.head<3>().norm() > 0.0003);
  if ((redo_ && imuMeasurements_.size() < 50) || redoCounter_ == 0) {
    redoPreintegration(speedAndBiases);
    redoCounter_++;
    Delta_b.setZero();
    redo_ = false;
  }
  const Eigen::Vector3d g_W = imuParameters_.g * Eigen::Vector3d(0, 0, 1);

  // propagated rotation and velocity at tw (first order in the bias change)
  const Eigen::Quaterniond q_WS_tw =
      q_WS * Delta_q_ * okvis::kinematics::deltaQ(-dalpha_db_g_ * Delta_b.head<3>());
  const Eigen::Matrix3d C_WS_tw = q_WS_tw.toRotationMatrix();
  const Eigen::Vector3d v_W_tw = speedAndBiases.head<3>() - g_W * Delta_t
      + C_WS_tk * (acc_integral_ + dv_db_g_ * Delta_b.head<3>() - C_integral_ * Delta_b.tail<3>());

  // Jacobian of the propagated (alpha, v) w.r.t. the minimal state at tk
  // [dr, dalpha, dv, db_g, db_a] (world-frame left perturbation, as in GpsErrorAsynchronous)
  Eigen::Matrix<double, 3, 15> Jalpha = Eigen::Matrix<double, 3, 15>::Zero();
  Jalpha.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
  // C_WS_tw = C_WS_tk * DeltaC * exp(-dalpha_db_g * db_g): a RIGHT (tw-body) perturbation;
  // as a world-frame left perturbation that is -C_WS_tw * dalpha_db_g (GpsErrorAsynchronous
  // uses C_WS_tk here, exact only for a small rotation over [tk, tw]).
  Jalpha.block<3, 3>(0, 9) = -C_WS_tw * dalpha_db_g_;
  Eigen::Matrix<double, 3, 15> Jv = Eigen::Matrix<double, 3, 15>::Zero();
  Jv.block<3, 3>(0, 3) = -okvis::kinematics::crossMx(C_WS_tk * acc_integral_);
  Jv.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity();
  Jv.block<3, 3>(0, 9) = C_WS_tk * dv_db_g_;
  Jv.block<3, 3>(0, 12) = -C_WS_tk * C_integral_;

  // ----- body-frame velocity and yaw rate -----
  const Eigen::Matrix3d C_BS = wheelParameters_.T_SB.C().transpose();
  const Eigen::Vector3d r_SB = wheelParameters_.T_SB.r();
  const Eigen::Vector3d omega_S = omega_S_tw_raw_ - speedAndBiases.segment<3>(3);
  const Eigen::Vector3d v_S = C_WS_tw.transpose() * v_W_tw;
  const Eigen::Vector3d v_B = C_BS * (v_S + omega_S.cross(r_SB));
  const Eigen::Vector3d omega_B = C_BS * omega_S;

  Eigen::Vector4d error;
  error.head<3>() = Eigen::Vector3d(measurement_[0], 0.0, 0.0) - v_B;
  error[3] = measurement_[1] - omega_B[2];
  error_ = error;
  const Eigen::Vector4d weighted = sqrtInformationDiag_.cwiseProduct(error);
  for (int i = 0; i < 4; ++i) residuals[i] = weighted[i];

  if (jacobians != nullptr) {
    // d v_S / d(alpha_tw, v_tw): (exp(da) C)^T v = C^T v + C^T [v]x da
    Eigen::Matrix<double, 3, 15> J_vS = C_WS_tw.transpose() * okvis::kinematics::crossMx(v_W_tw) * Jalpha
        + C_WS_tw.transpose() * Jv;
    // d (omega_S x r_SB) / d b_g = [r_SB]x  (omega_S = raw - b_g)
    J_vS.block<3, 3>(0, 9) += okvis::kinematics::crossMx(r_SB);
    Eigen::Matrix<double, 4, 15> Jmin = Eigen::Matrix<double, 4, 15>::Zero();
    Jmin.topRows<3>() = -C_BS * J_vS;
    Jmin.block<1, 3>(3, 9) = C_BS.row(2);  // d(-[C_BS (raw - b_g)]_z)/d b_g
    Jmin = sqrtInformationDiag_.asDiagonal() * Jmin;

    if (jacobians[0] != nullptr) {
      Eigen::Matrix<double, 6, 7, Eigen::RowMajor> lift;
      PoseManifold::minusJacobian(parameters[0], lift.data());
      Eigen::Map<Eigen::Matrix<double, 4, 7, Eigen::RowMajor>> J0(jacobians[0]);
      J0 = Jmin.leftCols<6>() * lift;
      if (jacobiansMinimal != nullptr && jacobiansMinimal[0] != nullptr) {
        Eigen::Map<Eigen::Matrix<double, 4, 6, Eigen::RowMajor>> J0min(jacobiansMinimal[0]);
        J0min = Jmin.leftCols<6>();
      }
    }
    if (jacobians[1] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 4, 9, Eigen::RowMajor>> J1(jacobians[1]);
      J1 = Jmin.rightCols<9>();
      if (jacobiansMinimal != nullptr && jacobiansMinimal[1] != nullptr) {
        Eigen::Map<Eigen::Matrix<double, 4, 9, Eigen::RowMajor>> J1min(jacobiansMinimal[1]);
        J1min = J1;
      }
    }
  }
  return true;
}

int WheelOdometryError::redoPreintegration(const okvis::SpeedAndBias& speedAndBiases) const {
  okvis::Time time = tk_;
  const okvis::Time end = tw_;
  Delta_q_ = Eigen::Quaterniond(1, 0, 0, 0);
  C_integral_.setZero();
  acc_integral_.setZero();
  cross_.setZero();
  dalpha_db_g_.setZero();
  dv_db_g_.setZero();
  speedAndBiases_ref_ = speedAndBiases;
  if (imuMeasurements_.size() < 2 || imuMeasurements_.front().timeStamp > time
      || imuMeasurements_.back().timeStamp < end) {
    return -1;  // not covered: zero increments, i.e. tw == tk behaviour
  }

  bool hasStarted = false;
  int i = 0;
  for (auto it = imuMeasurements_.begin(); it != imuMeasurements_.end(); ++it) {
    Eigen::Vector3d omega_S_0 = it->measurement.gyroscopes;
    Eigen::Vector3d acc_S_0 = it->measurement.accelerometers;
    Eigen::Vector3d omega_S_1 = (it + 1)->measurement.gyroscopes;
    Eigen::Vector3d acc_S_1 = (it + 1)->measurement.accelerometers;

    okvis::Time nexttime;
    if ((it + 1) == imuMeasurements_.end()) {
      nexttime = tw_;
    } else {
      nexttime = (it + 1)->timeStamp;
    }
    double dt = (nexttime - time).toSec();
    if (end < nexttime) {
      const double interval = (nexttime - it->timeStamp).toSec();
      nexttime = tw_;
      dt = (nexttime - time).toSec();
      const double r = dt / interval;
      omega_S_1 = ((1.0 - r) * omega_S_0 + r * omega_S_1).eval();
      acc_S_1 = ((1.0 - r) * acc_S_0 + r * acc_S_1).eval();
    }
    if (dt <= 0.0) continue;
    if (!hasStarted) {
      hasStarted = true;
      const double r = dt / (nexttime - it->timeStamp).toSec();
      omega_S_0 = (r * omega_S_0 + (1.0 - r) * omega_S_1).eval();
      acc_S_0 = (r * acc_S_0 + (1.0 - r) * acc_S_1).eval();
    }

    Eigen::Quaterniond dq;
    const Eigen::Vector3d omega_S_true = (0.5 * (omega_S_0 + omega_S_1) - speedAndBiases.segment<3>(3));
    const double theta_half = omega_S_true.norm() * 0.5 * dt;
    dq.vec() = okvis::kinematics::sinc(theta_half) * omega_S_true * 0.5 * dt;
    dq.w() = cos(theta_half);
    const Eigen::Quaterniond Delta_q_1 = Delta_q_ * dq;
    const Eigen::Matrix3d C = Delta_q_.toRotationMatrix();
    const Eigen::Matrix3d C_1 = Delta_q_1.toRotationMatrix();
    const Eigen::Vector3d acc_S_true = (0.5 * (acc_S_0 + acc_S_1) - speedAndBiases.segment<3>(6));
    const Eigen::Matrix3d C_integral_1 = C_integral_ + 0.5 * (C + C_1) * dt;
    const Eigen::Vector3d acc_integral_1 = acc_integral_ + 0.5 * (C + C_1) * acc_S_true * dt;

    dalpha_db_g_ += C_1 * okvis::kinematics::rightJacobian(omega_S_true * dt) * dt;
    const Eigen::Matrix3d cross_1 = dq.inverse().toRotationMatrix() * cross_
        + okvis::kinematics::rightJacobian(omega_S_true * dt) * dt;
    const Eigen::Matrix3d acc_S_x = okvis::kinematics::crossMx(acc_S_true);
    const Eigen::Matrix3d dv_db_g_1 = dv_db_g_ + 0.5 * dt * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);

    Delta_q_ = Delta_q_1;
    C_integral_ = C_integral_1;
    acc_integral_ = acc_integral_1;
    cross_ = cross_1;
    dv_db_g_ = dv_db_g_1;
    time = nexttime;
    ++i;
    if (nexttime == tw_) break;
  }
  return i;
}

}  // namespace ceres
}  // namespace okvis
