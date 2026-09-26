/**
 * mow-e (T-0125, ADR-0042 design item 3 / issue 7): wheel-encoder odometry factor.
 *
 * Body-frame velocity + yaw-rate residual on the state preceding the measurement,
 * IMU-preintegrated from the state time tk to the wheel time tw exactly like
 * GpsErrorAsynchronous, so that ViGraphEstimator::eliminateStateByImuMerge can
 * re-anchor the factor to an earlier state without losing it.
 *
 *   e_v = [v_x^enc, 0, 0]^T - C_BS * (C_WS(tw)^T * v_W(tw) + omega_S x r_SB)
 *   e_w = omega_z^enc - [C_BS * (omega_S^gyro(tw) - b_g)]_z
 *
 * with T_SB the body frame B in the IMU frame S (wheel_parameters.T_SB), sigma per axis
 * (sigma_v, sigma_lat, sigma_vert, sigma_omega). The IMU preintegration covariance is
 * not folded into the whitening (ponytail: the encoder sigmas dominate over the
 * <= keyframe-interval propagation; add it like GpsErrorAsynchronous::useImuCovariance
 * if bias-driven propagation error ever shows in the residual histogram).
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

#ifndef INCLUDE_OKVIS_CERES_WHEELODOMETRYERROR_HPP_
#define INCLUDE_OKVIS_CERES_WHEELODOMETRYERROR_HPP_

#include <mutex>
#include <ceres/sized_cost_function.h>

#include <okvis/FrameTypedefs.hpp>
#include <okvis/Time.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/ceres/ErrorInterface.hpp>

namespace okvis {
namespace ceres {

/// \brief Wheel odometry factor on (T_WS, SpeedAndBias) at tk, measurement at tw >= tk.
class WheelOdometryError :
    public ::ceres::SizedCostFunction<4 /* residuals: v_B xyz, omega_B z */,
        7 /* PoseParameterBlock T_WS at tk */,
        9 /* SpeedAndBiasParameterBlock at tk */>,
    public ErrorInterface {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef ::ceres::SizedCostFunction<4, 7, 9> base_t;
  static const int kNumResiduals = 4;
  typedef Eigen::Vector2d measurement_t;  ///< (v_x^enc [m/s], omega_z^enc [rad/s]).
  typedef Eigen::Vector4d sigmas_t;       ///< (sigma_v, sigma_lat, sigma_vert, sigma_omega).

  /// \brief Construct.
  /// @param measurement    (forward speed [m/s], yaw rate [rad/s]) in B.
  /// @param sigmas         Per-residual standard deviations (already inflated by any gate).
  /// @param imuMeasurements IMU deque spanning [tk, tw] (also provides the gyro at tw).
  /// @param imuParameters  IMU parameters.
  /// @param tk             State time.
  /// @param tw             Wheel measurement time (>= tk).
  /// @param wheelParameters wheel_parameters (T_SB used here).
  WheelOdometryError(const measurement_t& measurement, const sigmas_t& sigmas,
                     const okvis::ImuMeasurementDeque& imuMeasurements,
                     const okvis::ImuParameters& imuParameters,
                     const okvis::Time& tk, const okvis::Time& tw,
                     const okvis::WheelParameters& wheelParameters);

  virtual ~WheelOdometryError() {}

  /// \brief Raw gyro reading interpolated at time t (clamped to the deque's ends).
  static Eigen::Vector3d gyroAt(const okvis::ImuMeasurementDeque& imuMeasurements,
                                const okvis::Time& t);

  // getters
  const measurement_t& measurement() const { return measurement_; }
  const sigmas_t& sigmas() const { return sigmas_; }
  okvis::Time tk() const { return tk_; }
  okvis::Time tw() const { return tw_; }
  const okvis::ImuParameters& imuParameters() const { return imuParameters_; }
  const okvis::ImuMeasurementDeque& imuMeasurements() const { return imuMeasurements_; }
  const okvis::WheelParameters& wheelParameters() const { return wheelParameters_; }
  const Eigen::Vector3d& gyroAtTw() const { return omega_S_tw_raw_; }
  /// \brief Unweighted error of the last Evaluate().
  const Eigen::Vector4d& error() const { return error_; }

  virtual bool Evaluate(double const* const* parameters, double* residuals,
                        double** jacobians) const;
  virtual bool EvaluateWithMinimalJacobians(double const* const* parameters, double* residuals,
                                            double** jacobians, double** jacobiansMinimal) const;

  /// \brief Preintegrate the IMU from tk to tw at the given biases (bias linearisation point).
  int redoPreintegration(const okvis::SpeedAndBias& speedAndBiases) const;

  int residualDim() const { return kNumResiduals; }
  int parameterBlocks() const { return int(parameter_block_sizes().size()); }
  int parameterBlockDim(int parameterBlockId) const {
    return base_t::parameter_block_sizes().at(size_t(parameterBlockId));
  }
  virtual std::string typeInfo() const { return "WheelOdometryError"; }

 protected:
  measurement_t measurement_;
  sigmas_t sigmas_;
  Eigen::Vector4d sqrtInformationDiag_;
  mutable Eigen::Vector4d error_ = Eigen::Vector4d::Zero();
  okvis::WheelParameters wheelParameters_;
  okvis::ImuParameters imuParameters_;
  okvis::ImuMeasurementDeque imuMeasurements_;
  okvis::Time tk_;
  okvis::Time tw_;
  Eigen::Vector3d omega_S_tw_raw_;  ///< Gyro at tw (bias not removed).

  // ----- preintegration (mutable, as in GpsErrorAsynchronous) -----
  mutable std::mutex preintegrationMutex_;
  mutable Eigen::Quaterniond Delta_q_ = Eigen::Quaterniond(1, 0, 0, 0);
  mutable Eigen::Matrix3d C_integral_ = Eigen::Matrix3d::Zero();
  mutable Eigen::Vector3d acc_integral_ = Eigen::Vector3d::Zero();
  mutable Eigen::Matrix3d cross_ = Eigen::Matrix3d::Zero();
  mutable Eigen::Matrix3d dalpha_db_g_ = Eigen::Matrix3d::Zero();
  mutable Eigen::Matrix3d dv_db_g_ = Eigen::Matrix3d::Zero();
  mutable SpeedAndBias speedAndBiases_ref_ = SpeedAndBias::Zero();
  mutable bool redo_ = true;
  mutable int redoCounter_ = 0;
};

}  // namespace ceres
}  // namespace okvis

#endif  // INCLUDE_OKVIS_CERES_WHEELODOMETRYERROR_HPP_
