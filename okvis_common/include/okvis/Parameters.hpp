/**
 * OKVIS2-X - Open Keyframe-based Visual-Inertial SLAM Configurable with Dense 
 * Depth or LiDAR, and GNSS
 *
 * Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 * Copyright (c) 2020, Smart Robotics Lab / Imperial College London
 * Copyright (c) 2025, Mobile Robotics Lab / Technical University of Munich 
 * and ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

/**
 * @file Parameters.hpp
 * @brief This file contains struct definitions that encapsulate parameters and settings.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#ifndef INCLUDE_OKVIS_PARAMETERS_HPP_
#define INCLUDE_OKVIS_PARAMETERS_HPP_

#include <set>
#include <string>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#include <opencv2/core.hpp>
#pragma GCC diagnostic pop
#include <Eigen/Dense>
#include <okvis/Time.hpp>
#include <okvis/cameras/NCameraSystem.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <optional>

/// \brief okvis Main namespace of this package.
namespace okvis {

/// @brief Struct that contains all the camera calibration information.
struct CameraCalibration {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  okvis::kinematics::Transformation T_SC;   ///< Transformation from camera to sensor (IMU) frame.
  Eigen::Vector2i imageDimension;           ///< Image dimension. [pixels]
  Eigen::VectorXd distortionCoefficients;   ///< Distortion Coefficients.
  Eigen::Vector2d focalLength;              ///< Focal length.
  Eigen::Vector2d principalPoint;           ///< Principal point.
  std::string cameraModel;                  ///< camera model. ('pinhole' 'eucm')
  Eigen::Vector2d eucmParameters;           ///< alpha, beta eucm parameters

  /// \brief Distortion type. ('radialtangential' 'radialtangential8' 'equdistant')
  std::string distortionType;

  cameras::NCameraSystem::CameraType cameraType; ///< Some additional info about the camera.
};

/*!
 * \brief Camera parameters.
 *
 * A simple struct to specify properties of a Camera.
 *
 */
struct CameraParameters{
  double timestamp_tolerance; ///< Stereo frame out-of-sync tolerance. [s]
  std::set<size_t> sync_cameras; ///< The cameras that will be synchronised.
  std::vector<size_t> stereo_indices; ///< camera indices for the left and right camera for the stereo network

  /// \brief Image timestamp error. [s] timestamp_camera_correct = timestamp_camera - image_delay.
  double image_delay;

  /**
   * @brief Some parameters to set the online calibrator.
   */
  struct OnlineCalibrationParameters {
    bool do_extrinsics; ///< Do we online-calibrate extrinsics?
    bool do_extrinsics_final_ba; ///< Do we calibrate extrinsics in final BA?
    double sigma_r; ///< T_SCi position prior stdev [m]
    double sigma_alpha; ///< T_SCi orientation prior stdev [rad]
    double sigma_r_final_ba; ///< T_SCi position prior stdev in final BA [m]
    double sigma_alpha_final_ba; ///< T_SCi orientation prior stdev in final BA [rad]
  };

  OnlineCalibrationParameters online_calibration; ///< Online calibration parameters.
};

/*!
 * \brief IMU parameters.
 *
 * A simple struct to specify properties of an IMU.
 *
 */
struct ImuParameters{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool use; ///< Use the IMU at all?
  okvis::kinematics::Transformation T_BS; ///< Transform from Body frame to IMU (sensor frame S).
  double a_max;  ///< Accelerometer saturation. [m/s^2]
  double g_max;  ///< Gyroscope saturation. [rad/s]
  double sigma_g_c;  ///< Gyroscope noise density.
  double sigma_bg;  ///< Initial gyroscope bias.
  double sigma_a_c;  ///< Accelerometer noise density.
  double sigma_ba;  ///< Initial accelerometer bias
  double sigma_gw_c; ///< Gyroscope drift noise density.
  double sigma_aw_c; ///< Accelerometer drift noise density.
  Eigen::Vector3d g0;  ///< Mean of the prior gyro bias.
  Eigen::Vector3d a0;  ///< Mean of the prior accelerometer bias.
  double g;  ///< Earth acceleration.
  Eigen::Vector3d s_a; ///< Scale factor for accelerometer measurements
};

/**
 * @brief XFeat-on-TensorRT frontend parameters (Mow-e, ADR-0040). When
 *        `use` is set, detection/description runs XFeat instead of BRISK and
 *        descriptor matching switches to the float (cosine) metric; then
 *        `FrontendParameters::matching_threshold` is interpreted as a cosine
 *        DISTANCE (1 - cos similarity, unit descriptors: 0..2) instead of a
 *        BRISK Hamming distance, and the BRISK detection thresholds/octaves
 *        are ignored. Requires a build with USE_MOWE_XFEAT=ON.
 */
struct XFeatParameters {
  bool use = false; ///< Run the XFeat TensorRT frontend instead of BRISK.
  std::string engine; ///< Path to the XFeat .plan (static mono engine).
  std::string lighterglue_engine; ///< Path to the LighterGlue .plan ("" = cosine NN only).
  double score_threshold = 0.05; ///< Keypointness x reliability floor (upstream default).
  double keypoint_size = 16.0; ///< Nominal keypoint size [px]; sets obs. sigma = size/f*0.125.
  /// T-0129: matchToMap ratio test on L2 descriptor distance, best landmark vs second-best
  /// other landmark inside the keypoint's gate; 1.0 = off (optional yaml key map_ratio).
  double map_ratio = 1.0;
  double match_score_min = 0.10; ///< Min LighterGlue mutual-NN score to accept a match.
  int motion_stereo_top_n = 1; ///< Use LighterGlue for this many best-overlap motion-stereo frames.
  int stereo_min_nn_matches = 40; ///< matchStereo: run LighterGlue only when mutual-NN found fewer matches (T-0114).
};

/**
 * @brief VPR loop-closure parameters (Mow-e T-0120, mowe-nav-kb 06 §adapter/§aliasing,
 *        ADR-0042): DINOv2+VLAD retrieval replaces DBoW2 on the float-descriptor path.
 *        Candidate chain: retrieval (top_k, score_min) -> Mahalanobis prior gate ->
 *        LighterGlue match -> OpenGV GP3P RANSAC (min_inliers, reproj_px) -> temporal
 *        consistency (consecutive_required) -> ViSlamBackend::attemptLoopClosure.
 */
struct VprLoopParameters {
  std::string engine; ///< DINOv2 .plan (tools/onnx/export_dinov2.py); "" = VPR loop closure off.
  std::string vocabulary; ///< mowe_vpr MOWEVOC2 vocabulary (k-means + PCA whitening).
  int top_k = 3; ///< Retrieval candidates per keyframe query.
  double score_min = 0.40; ///< Min whitened-VLAD cosine to consider a candidate (T-0119 calibration).
  double prior_gate_sigma = 3.0; ///< Reject candidates beyond this many sigmas of the pose prior.
  double prior_sigma_pos_m = 0.3; ///< Position sigma floor of the prior [m].
  double prior_drift_frac = 0.0135; ///< Position sigma grows by this fraction of the path length between the frames.
  double prior_sigma_rot_deg = 30.0; ///< Orientation sigma of the prior [deg].
  int min_inliers = 20; ///< Min RANSAC (and refined) inliers.
  double reproj_px = 2.0; ///< RANSAC reprojection threshold [px].
  double min_inlier_ratio = 0.4; ///< Min RANSAC/refined inliers over LighterGlue landmark correspondences (BRISK path keeps upstream's 0.7).
  int consecutive_required = 2; ///< Verified candidates from this many consecutive keyframe queries must agree.
  int consecutive_max_gap = 5; ///< ...within this many database keyframes of each other.
};

/**
 * @brief Parameters for detection etc.
 */
struct FrontendParameters {
  double detection_threshold; ///< Detection threshold. By default the uniformity radius in pixels.
  double absolute_threshold; ///< Absolute Harris corner threshold (noise floor).
  double matching_threshold; ///< Descriptor matching threshold (BRISK Hamming; cosine distance with xfeat.use).
  int octaves; ///< Number of octaves for detection. 0 means single-scale at highest resolution.
  int max_num_keypoints; ///< Restrict to a maximum of this many keypoints per img (strongest ones).
  double keyframe_overlap; ///< Minimum field-of-view overlap.
  bool use_cnn; ///< Use the CNN (if available) to filter out dynamic content / sky.
  bool parallelise_detection; ///< Run parallel detect & describe.
  int num_matching_threads; ///< Parallelise matching with this number of threads.
  std::string detector = "brisk"; ///< Binary front-end: brisk | orb (Mow-e T-0134, optional key).
  XFeatParameters xfeat; ///< XFeat/LighterGlue frontend (Mow-e ADR-0040); off by default.
  VprLoopParameters vpr; ///< VPR loop closure (Mow-e T-0120); off unless vpr.engine is set.
};

/**
 * @brief Parameters regarding the estimator.
 */
struct EstimatorParameters {
  int num_keyframes; ///< Number of keyframes in optimisation window.
  int num_loop_closure_frames; ///< Number of loop closure frames in optimisation window.
  int num_imu_frames; ///< Number of frames linked by most recent nonlinear IMU error terms.
  bool do_loop_closures; ///< Whether to do VI-SLAM or VIO.
  bool do_final_ba; ///< Whether to run a final full BA.
  /// mow-e (T-0140): NO-OP. The wall-clock CeresIterationCallback is gone: its
  /// realtime_min_iterations floor (15-25 ms per iteration on the Orin Nano) defeated the
  /// budget, and a time-based stop is not reproducible. The realtime solve now stops at
  /// realtime_max_iterations or on the Ceres tolerances below, identically live and in replay.
  bool enforce_realtime;
  int realtime_min_iterations; ///< Unused since T-0140 (see enforce_realtime).
  int realtime_max_iterations; ///< Never do more than these, even if not converged.
  double realtime_time_limit; ///< Unused since T-0140 (see enforce_realtime). [s]
  /// mow-e (T-0140): deterministic early stop of the realtime solve (Ceres defaults if absent).
  double realtime_function_tolerance = 1e-6;
  double realtime_parameter_tolerance = 1e-8;
  /// mow-e (T-0140): observation budget per image of the newest frame, 0 = off. Surplus
  /// observations (grid-bucketed, ObservationBudget.hpp) are removed after data association.
  int realtime_max_observations_per_image = 0;
  /// mow-e (T-0141): non-keyframes get a newest-state-only realtime solve (landmarks, older
  /// states and T_GW constant; newest pose, speed and biases variable) on a sub-problem of the
  /// residuals touching it; keyframes keep the full window solve. Keyframe-ness is decided by
  /// matching before the solve, so this adds no timing dependence.
  bool realtime_cheap_on_nonkeyframes = false;
  /// mow-e (T-0141): iterations of the keyframe (full window) / non-keyframe (newest state)
  /// realtime solve when realtime_cheap_on_nonkeyframes is on; 0 = realtime_max_iterations.
  int realtime_keyframe_max_iterations = 0;
  int realtime_nonkeyframe_max_iterations = 0;
  /// mow-e (T-0141): full solves on every frame until this many keyframes were inserted.
  int realtime_cheap_warmup_keyframes = 0;
  /// mow-e (T-0141): middle tier - the non-keyframe solve keeps landmarks variable (older states constant).
  bool realtime_cheap_landmarks_variable = false;
  /// mow-e (T-0141): newest states left variable by the non-keyframe solve (1 = newest only).
  int realtime_cheap_variable_states = 1;
  /// mow-e (T-0141): live (non-blocking) camera input queue depth in frames; the oldest frame is
  /// dropped when full. Upstream 2 (100 ms at 20 Hz) drops frames on single solve spikes.
  int camera_input_queue_size = 2;
  int realtime_num_threads; ///< Number of threads for the realtime optimisation.
  int full_graph_iterations; ///< Don't do more than these for the full (background) optimisation.
  int full_graph_num_threads; ///< Number of threads for the full (background) optimisation.
  /// mow-e (T-0131): blocking (synchronous replay) mode only. < 0 = live behaviour: the
  /// full-graph result is imported whenever the background thread happens to finish.
  /// N >= 0: the thread is joined N frames after its launch and imported in that frame,
  /// solved with 1 thread, so the loop-closure import lands at a fixed frame index.
  int full_graph_join_frames = -1;
  double p_dbow; ///< Match threshold for dBoW -- unfortunately this varies with setups.
  double drift_percentage_heuristic; ///< % allowed drift in loop closures rel. to dist. travelled.
};

/**
 * @brief Some options for how and what to output.
 */
struct OutputParameters {
    bool display_topview; ///< Displays top view (Non-causal part might be slow).
    bool display_matches; ///< Displays debug video and matches. May be slow.
    bool display_overhead; ///< Debug overhead image. Is slow.
    bool enable_submapping; //< Whether or not is submapping enabled
};
/**
  * @brief Struct to specify parameters of GPS sensor
  */
struct GpsParameters {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    std::string type; ///< Format of GPS data: "cartesian" | "geodetic" | "geodetic-leica":
    Eigen::Vector3d r_SA; ///< Translation IMU sensor to GPS antenna; known from calibration
    double yawErrorThreshold; /// < Threshold on maximum estimated yaw error [degree] for initialization
    bool robustGpsInit; /// < Flag if robust initialization is needed (low-grade GPS sensor)

    /// Default Constructor (no GPS)
    GpsParameters() : type("none"), r_SA(Eigen::Vector3d(0., 0., 0.)),
                      yawErrorThreshold(0.), robustGpsInit(false)
                      {}
};


/**
 * @brief mow-e (T-0125, ADR-0042 design item 3): wheel-encoder odometry parameters
 *        (config block `wheel_parameters:`; names fixed by shared/contracts/wheel_odometry.md).
 */
struct WheelParameters {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    okvis::kinematics::Transformation T_SB; ///< Body frame B (base_link) in the IMU frame S.
    double sigma_v = 0.05;       ///< Forward-speed sigma [m/s].
    double sigma_lat = 0.5;      ///< Lateral (no-side-slip) sigma [m/s]; large where the planar assumption fails.
    double sigma_vert = 0.5;     ///< Vertical sigma [m/s].
    double sigma_omega = 0.05;   ///< Yaw-rate sigma [rad/s].
    double b_eff = 0.5;          ///< Effective track width [m] (calibrated, see the contract).
    double slip_gate_omega = 0.2; ///< |omega_enc - omega_gyro| above this [rad/s] -> factor skipped.
    double slip_gate_v = 0.3;    ///< |v_enc - v_B,x(state)| above this [m/s] -> sigma_v inflated.
    std::string loss = "cauchy"; ///< Robust loss: "cauchy" | "huber".
};

/// @brief  Struct to specify the parameters of a LiDAR sensor
struct LidarParameters {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    okvis::kinematics::Transformation T_SL;   ///< Transformation from LiDAR to sensor (IMU) frame.
    float elevation_resolution_angle; ///< Resolution angle for the elevation of the LiDAR sensor
    float azimuth_resolution_angle; ///< Resolution angle for the azimuth of the LiDAR sensor
};


/// @brief Struct to combine all parameters and settings.
struct ViParameters {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  okvis::cameras::NCameraSystem nCameraSystem;  ///< Camera extrinsics and intrinsics.
  CameraParameters camera; ///< Camera parameters.
  ImuParameters imu; ///< Imu parameters.
  std::optional<GpsParameters> gps; ///< Gps parameters.
  std::optional<WheelParameters> wheel; ///< mow-e (T-0125): wheel odometry parameters.
  std::optional<LidarParameters> lidar; ///< LiDAR parameters
  FrontendParameters frontend; ///< Frontend parameters.
  EstimatorParameters estimator; ///< Estimator parameters.
  OutputParameters output; ///< Output parameters.
  CameraCalibration rgb;  ///< RGB parameters.
};

} // namespace okvis

#endif // INCLUDE_OKVIS_PARAMETERS_HPP_
