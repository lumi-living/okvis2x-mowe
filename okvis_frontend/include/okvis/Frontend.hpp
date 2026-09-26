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
 * @file Frontend.hpp
 * @brief Header file for the Frontend class.
 * @author Andreas Forster
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_FRONTEND_HPP_
#define INCLUDE_OKVIS_FRONTEND_HPP_

#include <mutex>

#include <okvis/Component.hpp>
#include <okvis/ViFrontendInterface.hpp>
#include <okvis/ViSlamBackend.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/timing/Timer.hpp>
#include <thread>

/// \brief okvis Main namespace of this package.
namespace okvis {

/**
 * @brief A frontend using BRISK features
 */
class Frontend : public ViFrontendInterface {
 public:
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief Constructor.
   * @param numCameras Number of cameras in the sensor configuration.
   * @param dBowVocDir The directory containing the DBoW vocabulary.
   */
  Frontend(size_t numCameras, std::string dBowVocDir);
  virtual ~Frontend() override;

  /**
   * @brief Load another, previously saved VI-SLAM component.
   * @param filename Filename.
   * @param imuParameters Imu parameters of the loaded component.
   * @param nCameraSystem Multi-camera configuration of the loaded component.
   * @param componentFixed Should the component be treated as fixed?
   * @return True on success.
   */
  bool loadComponent(std::string filename,
                     const ImuParameters &imuParameters,
                     const cameras::NCameraSystem &nCameraSystem,
                     bool componentFixed = true);

  ///@{
  /**
   * @brief Detection and descriptor extraction on a per image basis.
   * @remark This method is threadsafe.
   * @param cameraIndex Index of camera to do detection and description.
   * @param frameOut    Multiframe containing the frames.
   *                    Resulting keypoints and descriptors are saved in here.
   * @param T_WC        Pose of camera with index cameraIndex at image capture time.
   * @param[in] keypoints If the keypoints are already available from a different source, provide
   *                      them here in order to skip detection.
   * @warning Using keypoints from a different source is not yet implemented.
   * @return True if successful.
   */
  virtual bool detectAndDescribe(size_t cameraIndex,
                                 std::shared_ptr<okvis::MultiFrame> frameOut,
                                 const okvis::kinematics::Transformation& T_WC,
                                 const std::vector<cv::KeyPoint> * keypoints) override final;

  /**
   * @brief Matching as well as initialization of landmarks and state.
   * @warning This method is not threadsafe.
   * @warning This method uses the estimator. Make sure to not access it in another thread.
   * @param estimator       Estimator.
   * @param params          Configuration parameters.
   * @param framesInOut     Multiframe including the descriptors of all the keypoints.
   * @param kfPrior         A prior to trigger new keyframe from other criteria (e.g. LiDAR overlap)
   * @param[out] asKeyframe Should the frame be a keyframe?
   * @return True if successful.
   */
  virtual bool dataAssociationAndInitialization(
      Estimator& estimator,
      const okvis::ViParameters & params,
      std::shared_ptr<okvis::MultiFrame> framesInOut, bool kfPrior, bool* asKeyframe) override final;

  /**
   * @brief Propagates pose, speeds and biases with given IMU measurements.
   * @see okvis::ceres::ImuError::propagation()
   * @remark This method is threadsafe.
   * @param[in] imuMeasurements All the IMU measurements.
   * @param[in] imuParams The parameters to be used.
   * @param[inout] T_WS_propagated Start pose.
   * @param[inout] speedAndBiases Start speed and biases.
   * @param[in] t_start Start time.
   * @param[in] t_end End time.
   * @param[out] covariance Covariance for GIVEN start states.
   * @param[out] jacobian Jacobian w.r.t. start states.
   * @return True on success.
   */
  virtual bool propagation(const okvis::ImuMeasurementDeque & imuMeasurements,
                           const okvis::ImuParameters & imuParams,
                           okvis::kinematics::Transformation& T_WS_propagated,
                           okvis::SpeedAndBias & speedAndBiases,
                           const okvis::Time& t_start, const okvis::Time& t_end,
                           Eigen::Matrix<double, 15, 15>* covariance,
                           Eigen::Matrix<double, 15, 15>* jacobian) const override final;

  ///@}
  /// @name Getters related to the BRISK detector
  /// @{

  /// @brief Get the number of octaves of the BRISK detector.
  size_t getBriskDetectionOctaves() const {
    return briskDetectionOctaves_;
  }

  /// @brief Get the detection threshold of the BRISK detector.
  double getBriskDetectionThreshold() const {
    return briskDetectionThreshold_;
  }

  /// @brief Get the absolute threshold of the BRISK detector.
  double getBriskDetectionAbsoluteThreshold() const {
    return briskDetectionAbsoluteThreshold_;
  }

  /// @brief Get the maximum amount of keypoints of the BRISK detector.
  size_t getBriskDetectionMaximumKeypoints() const {
    return briskDetectionMaximumKeypoints_;
  }

  ///@}
  /// @name Getters related to the BRISK descriptor
  /// @{

  /// @brief Get the rotation invariance setting of the BRISK descriptor.
  bool getBriskDescriptionRotationInvariance() const {
    return briskDescriptionRotationInvariance_;
  }

  /// @brief Get the scale invariance setting of the BRISK descriptor.
  bool getBriskDescriptionScaleInvariance() const {
    return briskDescriptionScaleInvariance_;
  }

  ///@}
  /// @name Other getters
  /// @{

  /// @brief Get the matching threshold.
  double getBriskMatchingThreshold() const {
    return briskMatchingThreshold_;
  }

  /// @brief Get the area overlap threshold under which a new keyframe is inserted.
  float getKeyframeInsertionOverlapThershold() const {
    return keyframeInsertionOverlapThreshold_;
  }

  /// @brief Returns true if the initialization has been completed (RANSAC with actual translation)
  bool isInitialized() const {
    return isInitialized_;
  }

  /// @}
  /// @name Setters related to the BRISK detector
  /// @{

  /// @brief Set the number of octaves of the BRISK detector.
  void setBriskDetectionOctaves(size_t octaves) {
    briskDetectionOctaves_ = octaves;
    initialiseBriskFeatureDetectors();
  }

  /// @brief Set the detection threshold of the BRISK detector.
  void setBriskDetectionThreshold(double threshold) {
    briskDetectionThreshold_ = threshold;
    initialiseBriskFeatureDetectors();
  }

  /// @brief Set the absolute threshold of the BRISK detector.
  void setBriskDetectionAbsoluteThreshold(double threshold) {
    briskDetectionAbsoluteThreshold_ = threshold;
    initialiseBriskFeatureDetectors();
  }

  /// @brief Set the maximum number of keypoints of the BRISK detector.
  void setBriskDetectionMaximumKeypoints(size_t maxKeypoints) {
    briskDetectionMaximumKeypoints_ = maxKeypoints;
    initialiseBriskFeatureDetectors();
  }

  /// @}
  /// @name Setters related to the BRISK descriptor
  /// @{

  /// @brief Set the rotation invariance setting of the BRISK descriptor.
  void setBriskDescriptionRotationInvariance(bool invariance) {
    briskDescriptionRotationInvariance_ = invariance;
    initialiseBriskFeatureDetectors();
  }

  /// @brief Set the scale invariance setting of the BRISK descriptor.
  void setBriskDescriptionScaleInvariance(bool invariance) {
    briskDescriptionScaleInvariance_ = invariance;
    initialiseBriskFeatureDetectors();
  }

  ///@}
  /// @name Other setters
  /// @{

  /// @brief Set the matching threshold.
  void setBriskMatchingThreshold(double threshold) {
    briskMatchingThreshold_ = threshold;
  }

  /// @brief Set the area overlap threshold under which a new keyframe is inserted.
  void setKeyframeInsertionOverlapThreshold(float threshold) {
    keyframeInsertionOverlapThreshold_ = threshold;
  }

  /// @brief Enable/configure the XFeat-on-TensorRT frontend (Mow-e, ADR-0040).
  /// Loads the XFeat engine(s) (one per camera: detectAndDescribe runs
  /// per-camera in parallel and a TensorRT execution context is not
  /// thread-safe) and, when configured, the LighterGlue matcher engine.
  /// Throws when xfeat.use is set but the build lacks USE_MOWE_XFEAT.
  void setXFeatParameters(const XFeatParameters& xfeat);

  /// @brief Whether the XFeat frontend is active (loaded engines, use=true).
  bool usingXFeat() const;

  /// @brief Select the descriptor metric (Mow-e T-0112, functor seam): false =
  ///        BRISK 48-byte rows / Hamming, true = 64-D unit-norm float rows /
  ///        cosine distance. setXFeatParameters() sets this from xfeat.use;
  ///        exposed so descriptors injected by other means (tests, replay) can
  ///        use the float path without a TensorRT engine. Also gates every
  ///        DBoW2 path (BRISK vocabulary, unusable with float descriptors).
  void setFloatDescriptors(bool floatDescriptors) {
    floatDescriptors_ = floatDescriptors;
  }

  /// @brief Whether the float/cosine descriptor path is selected.
  bool floatDescriptors() const {
    return floatDescriptors_;
  }

  /// \brief Bytes per keypoint descriptor: 48 (BRISK) or 256 (XFeat 64 float).
  size_t descriptorBytes() const;

  /// \brief Front-end statistics (Mow-e T-0113), accumulated over the run and
  ///        written by the apps on exit as frontend_stats.json. Counts are per
  ///        okvis::Frame (one camera image); frontendMs is per MultiFrame.
  struct Stats {
    std::vector<double> frontendMs; ///< detect+describe wall time per MultiFrame [ms]
    uint64_t keypoints = 0;         ///< keypoints summed over camera frames
    uint64_t frames = 0;            ///< camera frames (images) detected on
    uint64_t stereoMatches = 0;     ///< accepted L-R stereo matches (matchStereo)
    uint64_t stereoCalls = 0;       ///< matchStereo pair evaluations (keyframes)
    uint64_t stereoLgFallbacks = 0; ///< stereo pairs where NN < stereo_min_nn_matches ran LighterGlue (T-0114)
    uint64_t keyframeMatches = 0;   ///< 3d2d matches to the map (matchToMap)
    uint64_t keyframeCalls = 0;     ///< matchToMap calls (every frame after the first)
    uint64_t loopClosures = 0;      ///< accepted loop closures
  };
  /// \brief Snapshot of the statistics.
  Stats stats() const;
  /// \brief Write stats() as JSON (mean/p99 of frontendMs, per-frame means,
  ///        engine_loaded flag). Returns false when the file cannot be written.
  bool writeStatsJson(const std::string& path) const;

  /// @}
  /// @name VPR loop closure (Mow-e T-0120, mowe-nav-kb 06 §adapter, ADR-0042)
  /// @{

  /// \brief Enable VPR loop closure on the float-descriptor path: loads the DINOv2
  ///        engine + VLAD vocabulary (vpr.engine empty = off). Throws when an engine
  ///        is configured but the build lacks USE_MOWE_XFEAT.
  void setVprLoopParameters(const VprLoopParameters& vpr);

  /// \brief Whether VPR loop closure is active (engine + vocabulary loaded).
  bool usingVprLoopClosure() const;

  /// \brief Load another run's keyframes.bin (VPR descriptors, XFeat features,
  ///        S-frame landmarks, poses) as a FOREIGN retrieval database: candidates
  ///        are verified and counted, never fed to the graph (seed of T-0121).
  bool loadForeignKeyframes(const std::string& path, const cameras::NCameraSystem& cameraSystem);

  /// \brief Write this run's VPR keyframe database as keyframes.bin.
  bool saveKeyframes(const Estimator& estimator, const std::string& path) const;

  /// \brief Loop-closure funnel counters (in-session and foreign-map), written by
  ///        the apps as loop_stats.json.
  struct LoopStats {
    uint64_t queries = 0;                 ///< keyframe VPR queries
    uint64_t candidates = 0;              ///< in-session candidates above score_min
    uint64_t candidatesSkippedState = 0;  ///< not a pose-graph / place-recognition frame, or a (recent) loop-closure frame
    uint64_t rejectedByPriorGate = 0;     ///< Mahalanobis prior gate
    uint64_t rejectedByGeometry = 0;      ///< LighterGlue + GP3P RANSAC + refinement
    uint64_t rejectedByTemporal = 0;      ///< temporal consistency
    uint64_t rejectedByEstimator = 0;     ///< ViSlamBackend::attemptLoopClosure refused
    uint64_t loopsAccepted = 0;
    uint64_t foreignCandidates = 0;
    uint64_t foreignRejectedByGeometry = 0;
    uint64_t foreignRejectedByTemporal = 0;
    uint64_t loopsAcceptedAgainstForeignMap = 0;
    /// per accepted foreign-map loop, so a run's evidence survives a truncated log
    /// (the runner keeps only `tail -30`): query stamp, map keyframe stamp, VPR
    /// score, verified |t_Sold_Snew| [m]. Classified against mocap by
    /// tools/eval/check_foreign_loops.py.
    struct ForeignLoop { uint64_t frameId, queryNs, keyframeId, keyframeNs; double score, tNormM; };
    std::vector<ForeignLoop> foreignLoops;
    std::vector<double> verificationMs;   ///< per verifyRecognisedPlace call (VPR path)
    std::vector<double> embedMs;          ///< per keyframe VPR descriptor (embed + VLAD)
    std::vector<double> priorMahalanobis; ///< per gated candidate
  };
  LoopStats loopStats() const;
  bool writeLoopStatsJson(const std::string& path) const;

  /// \brief Descriptor distance dispatch (okvis/DescriptorDistance.hpp): BRISK
  ///        Hamming popcount, or cosine distance (1 - dot, unit descriptors)
  ///        when floatDescriptors() is set. matching_threshold is interpreted
  ///        on the active scale.
  double descriptorDist(const unsigned char* a, const unsigned char* b) const;

  /// @}

  /// \brief Stop all CNN background threads.
  void endCnnThreads();

  /// \brief Clears and resets everything (so you can re-start).
  void clear();

private:

  /**
   * @brief   feature detectors with the current settings.
   *          The vector contains one for each camera to ensure that there are no problems with
   *          parallel detection.
   * @warning Lock with featureDetectorMutexes_[cameraIndex] when using the detector.
   */
  std::vector<std::shared_ptr<cv::FeatureDetector> > featureDetectors_;
  /**
   * @brief   feature descriptors with the current settings.
   *          The vector contains one for each camera to ensure that there are no problems with
   *          parallel detection.
   * @warning Lock with featureDetectorMutexes_[cameraIndex] when using the descriptor.
   */
  std::vector<std::shared_ptr<cv::DescriptorExtractor> > descriptorExtractors_;
  /// Mutexes for feature detectors and descriptors.
  std::vector<std::unique_ptr<std::mutex> > featureDetectorMutexes_;

  bool isInitialized_;        ///< Is the pose initialised?
  const size_t numCameras_;   ///< Number of cameras in the configuration.

  /// @name BRISK detection parameters
  /// @{

  size_t briskDetectionOctaves_;            ///< The set number of brisk octaves.
  double briskDetectionThreshold_;          ///< The set BRISK detection threshold.
  double briskDetectionAbsoluteThreshold_;  ///< The set BRISK absolute detection threshold.
  size_t briskDetectionMaximumKeypoints_;   ///< The set maximum number of keypoints.

  /// @}
  /// @name BRISK descriptor extractor parameters
  /// @{

  bool briskDescriptionRotationInvariance_; ///< The set rotation invariance setting.
  bool briskDescriptionScaleInvariance_;    ///< The set scale invariance setting.

  ///@}
  /// @name BRISK matching parameters
  ///@{

  double briskMatchingThreshold_; ///< Matching threshold (Hamming; cosine distance with XFeat).

  ///@}
  /// @name XFeat frontend (Mow-e, ADR-0040)
  ///@{

  XFeatParameters xfeatParams_; ///< XFeat/LighterGlue configuration (use=false by default).
  /// \brief Holds the per-camera XFeat engines (+ LighterGlue matcher, PIMPL —
  ///        keeps TensorRT types out of this header; only built with
  ///        USE_MOWE_XFEAT).
  struct XFeatRuntime;
  std::unique_ptr<XFeatRuntime> xfeatRuntime_;

  /// \brief VPR loop closure state (T-0120, PIMPL): embedder, vocabulary, in-session
  ///        + foreign databases, temporal windows. Null unless configured.
  struct VprLoop;
  std::unique_ptr<VprLoop> vprLoop_;
  VprLoopParameters vprParams_;
  LoopStats loopStats_;
  /// \brief Embed the left image of a keyframe into vprLoop_ (current descriptor).
  bool vprEmbedCurrent(const MultiFrame& frame);
  /// \brief In-session candidates above score_min (score-descending) for the current
  ///        descriptor; foreign-map candidates are verified + counted here.
  void vprQuery(const Estimator& estimator, const okvis::ViParameters& params,
                std::shared_ptr<okvis::MultiFrame> framesInOut,
                std::vector<std::pair<StateId, double>>& stateIds);
  /// \brief Add the current descriptor as a database keyframe.
  void vprAddCurrent(const Estimator& estimator, const MultiFrame& frame);

  /// \brief XFeat replacement for detect+describe: run the camera's engine on
  ///        the frame image, threshold + cap to max keypoints, and store
  ///        keypoints & float descriptors into the MultiFrame.
  bool detectAndDescribeXFeat(size_t cameraIndex,
                              std::shared_ptr<okvis::MultiFrame> frameOut);

  /// \brief LighterGlue pair proposals between (frameA, imA) and (frameB,
  ///        imB): fills matchBForA[kA] = matched kB or -1. Returns false when
  ///        LighterGlue is unavailable — caller falls back to brute-force
  ///        descriptor distance. (ADR-0040 stage B.)
  bool lighterGluePairProposals(const okvis::MultiFrame& frameA, size_t imA,
                                const okvis::MultiFrame& frameB, size_t imB,
                                std::vector<int>& matchBForA);

  bool floatDescriptors_ = false; ///< Descriptor metric: float/cosine (true) or BRISK/Hamming.

  mutable std::mutex statsMutex_; ///< Guards stats_ (detection threads + processing thread).
  Stats stats_;                   ///< Accumulated front-end statistics (T-0113).

  ///@}

  /**
   * @brief If the hull-area around all matched keypoints of the current frame (with existing
   *        landmarks)
   *        divided by the hull-area around all keypoints in the current frame is lower than
   *        this threshold it should be a new keyframe.
   * @see   doWeNeedANewKeyframe()
   */
  float keyframeInsertionOverlapThreshold_;  //0.55

  /**
   * @brief Decision whether a new frame should be keyframe or not, based on overlap heuristic.
   * @param estimator     const reference to the estimator.
   * @param currentFrame  Keyframe candidate.
   * @return True if it should be a new keyframe.
   */
  bool doWeNeedANewKeyframe(const Estimator& estimator,
                            std::shared_ptr<okvis::MultiFrame> currentFrame);

  /**
   * @brief Match a new multiframe to existing keyframes
   * @tparam MATCHING_ALGORITHM Algorithm to match new keypoints to existing landmarks
   * @warning As this function uses the estimator it is not threadsafe
   * @param      estimator              Estimator.
   * @param[in]  params                 Parameter struct.
   * @param[in]  currentFrameId         ID of the current frame that should be matched against
   *                                    keyframes.
   * @param[in]  loopClosureLandmarksToUseExclusively Use these landmarks exclusively, if supplied.
   * @return The number of matches in total.
   */
  template<class CAMERA_GEOMETRY>
  int matchToMap(Estimator& estimator,
                 const okvis::ViParameters& params,
                 const uint64_t currentFrameId,
                 const std::set<LandmarkId>* loopClosureLandmarksToUseExclusively = nullptr);

  /**
   * @brief Match the frames inside the multiframe to each other to initialise new landmarks.
   * @tparam MATCHING_ALGORITHM Algorithm to match new keypoints to existing landmarks.
   * @warning As this function uses the estimator it is not threadsafe.
   * @param estimator Estimator.
   * @param multiFrame Multiframe containing the frames to match.
   * @param params The VI parameters.
   * @param asKeyframe Whether this is a keyframe.
   */
  template<class CAMERA_GEOMETRY>
  void matchStereo(Estimator& estimator,
                   std::shared_ptr<okvis::MultiFrame> multiFrame,
                   const okvis::ViParameters& params,
                   bool asKeyframe);

  /// \brief DBoW for loop closure
  /// https://en.cppreference.com/w/cpp/language/pimpl
  class DBoW;
  std::string dBowVocDir_; ///< Vocabulary directory; loaded lazily by dBow().
  std::unique_ptr<DBoW> dBow_; ///< DBoW object (PIMPL); null until first BRISK use.
  /// \brief Lazily load the BRISK DBoW2 vocabulary (T-0112): never touched on
  ///        the float-descriptor path, so XFeat runs need no small_voc.yml.gz.
  DBoW& dBow();

  /**
   * @brief Get filtered DBoW query.
   * @param[in] dBow DBoW database to use.
   * @param[in] features Features to match against DBoW.
   * @param[out] stateIds Resulting matching (keyframe) pose IDs with corresponding scores.
   * @return Number of matching keyframes.
   */
  int getFilteredDBoWResult(const std::unique_ptr<DBoW> &dBow,
                            const std::vector<std::vector<uchar>> &features,
                            std::vector<std::pair<StateId, double>> &stateIds) const;

  /**
   * @brief Verifies a recognised place with 3D2D matching, ransac, and nonlinear pose refinement.
   * @param[in] estimator Estimator.
   * @param[in] params The VI parameters.
   * @param[in] framesInOut Current multiframe.
   * @param[in] oldFrame Old multiframe to match against.
   * @param[out] T_Sold_Snew The relative pose found.
   * @param[out] H Information matrix corresponding to T_Sold_Snew.
   * @param[in] minInliers Minimum number of inliers required.
   */
  bool verifyRecognisedPlace(const Estimator &estimator,
                             const okvis::ViParameters &params,
                             const std::shared_ptr<const MultiFrame> framesInOut,
                             const std::shared_ptr<const MultiFrame> oldFrame,
                             kinematics::Transformation &T_Sold_Snew,
                             Eigen::Matrix<double, 6, 6>& H,
                             int minInliers = 10);

  /**
   * @brief Perform 3D/2D RANSAC.
   * @warning As this function uses the estimator it is not threadsafe.
   * @param estimator       Estimator.
   * @param nCameraSystem   Camera configuration and parameters.
   * @param currentFrame    Frame with the new potential matches.
   * @param initializePose  Initialize the pose from RANSAC?
   * @param removeOutliers  Remove observation of outliers in estimator.
   * @return True on success.
   */
  bool runRansac3d2d(Estimator &estimator,
                    const okvis::cameras::NCameraSystem &nCameraSystem,
                    std::shared_ptr<okvis::MultiFrame> currentFrame,
                    bool initializePose,
                    bool removeOutliers);
  /**
   * @brief Remove outliers on current frame.
   * @warning As this function uses the estimator it is not threadsafe.
   * @param estimator       Estimator.
   * @param nCameraSystem   Camera configuration and parameters.
   * @param currentFrame    Frame with the new potential matches.
   * @return Number of inliers.
   */
  template <class CAMERA_GEOMETRY>
  int removeOutliers(Estimator &estimator,
                     const okvis::cameras::NCameraSystem &nCameraSystem,
                     std::shared_ptr<okvis::MultiFrame> currentFrame);

  /**
   * @brief Perform 2D/2D RANSAC.
   * @warning As this function uses the estimator it is not threadsafe.
   * @param estimator         Estimator.
   * @param params            Parameter struct.
   * @param currentFrameId    ID of the new multiframe containing matches with the frame with ID
   *                          olderFrameId.
   * @param olderFrameId      ID of the multiframe to which the current frame has been matched
   *                          against.
   * @param initializePose    If the pose has not yet been initialised should the function try to
   *                          initialise it.
   * @param removeOutliers    Remove observation of outliers in estimator.
   * @param[out] rotationOnly Was the rotation only RANSAC model enough to explain the matches.
   * @return Number of inliers.
   */
  int runRansac2d2d(Estimator& estimator,
                    const okvis::ViParameters& params, uint64_t currentFrameId,
                    uint64_t olderFrameId, bool initializePose,
                    bool removeOutliers, bool &rotationOnly);

  /// (re)instantiates feature detectors and descriptor extractors. Used after settings changed or
  /// at startup.
  void initialiseBriskFeatureDetectors();

  /// \brief Classification network for keypoints (if enabled).
  std::vector<std::shared_ptr<Network>> networks_;

  /**
   * @brief Match the frames to older, co-visible frames and triangulate.
   * @tparam CAMERA_GEOMETRY The camera geometry type to use.
   * @warning As this function uses the estimator it is not threadsafe.
   * @param estimator Estimator.
   * @param params The VI parameters.
   * @param currentFrameId Current frame ID.
   * @param[out] rotationOnly Result obtained matches with rotation-only RANSAC.
   */
  template<class CAMERA_GEOMETRY>
  int matchMotionStereo(Estimator &estimator, const okvis::ViParameters &params,
                        const uint64_t currentFrameId, bool &rotationOnly);

  /// \brief Helper struct (internally) for landmarks to match.
  struct LandmarkToMatch {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3d p_W; ///< 3D point in World coordinates.
    cv::Mat descriptors; ///< All its descriptors.
    std::vector<KeypointIdentifier> kids; ///< All its observations.
    Eigen::Matrix3Xd e_W; ///< All directions in W-coords of the observations.
    Eigen::Matrix3Xd r_W; ///< Image centres of all the observations (W-coords).
    bool is3d = false; ///< Determine whether treated as 3D initialised.
    Eigen::Vector2d projection; ///< 2D projection location in pixels.
    bool ignore = false; ///< Ignore if classified as sky / person.
  };

  /**
   * @brief Parallelisable sub-part of matchToMap -- proper 3D points..
   * @tparam CAMERA_GEOMETRY The camera geometry type to use.
   * @param threadIdx Thread index.
   * @param numThreads Total number of threads to use.
   * @param estimator Estimator.
   * @param params The VI parameters.
   * @param currentFrameId Current frame ID.
   * @param loopClosureLandmarksToUseExclusively Only use these, if not nullptr.
   * @param T_WS1 Pose guess.
   * @param landmarksToMatch Landmarks to be matched against.
   * @param numKeypoints Number of keypoints (in image im)
   * @param pointMap Current map.
   * @param im The image idx.
   * @param multiFrame current multiFrame.
   * @param[out] distances Distances of landmarks.
   * @param[out] lmIds matched landmark IDs.
   * @param[out] hps_W matched landmarks (homogeneous) positions.
   * @param[out] ctrs Number of matches (by im).
   * @param[out] reprErrs Reprojection errors (by im).
   */
  template<class CAMERA_GEOMETRY>
  void matchToMapByThread(
      size_t threadIdx, size_t numThreads, const Estimator &estimator,
      const okvis::ViParameters& params, const uint64_t currentFrameId,
      const std::set<LandmarkId>* loopClosureLandmarksToUseExclusively,
      const kinematics::Transformation& T_WS1,
      const AlignedMap<LandmarkId, LandmarkToMatch>& landmarksToMatch,
      size_t numKeypoints,
      const MapPoints& pointMap, size_t im, const MultiFramePtr&  multiFrame,
      std::vector<double>& distances, std::vector<LandmarkId>& lmIds,
      AlignedVector<Eigen::Vector4d>& hps_W, std::vector<size_t>& ctrs,
      std::vector<double>& reprErrs) const;

  /**
   * @brief Parallelisable sub-part of matchToMap -- unitialised points.
   * @tparam CAMERA_GEOMETRY The camera geometry type to use.
   * @param threadIdx Thread index.
   * @param numThreads Total number of threads to use.
   * @param estimator Estimator.
   * @param params The VI parameters.
   * @param currentFrameId Current frame ID.
   * @param loopClosureLandmarksToUseExclusively Only use these, if not nullptr.
   * @param T_WS1 Pose guess.
   * @param landmarksToMatch Landmarks to be matched against.
   * @param numKeypoints Number of keypoints (in image im)
   * @param pointMap Current map.
   * @param im The image idx.
   * @param multiFrame current multiFrame.
   * @param[out] distances Distances of landmarks.
   * @param[out] lmIds matched landmark IDs.
   * @param[out] hps_W matched landmarks (homogeneous) positions.
   * @param[out] ctrs Number of matches (by im).
   */
  template<class CAMERA_GEOMETRY>
  void matchToMapByThreadUnitialised(
      size_t threadIdx, size_t numThreads, const Estimator &estimator,
      const okvis::ViParameters& params, const uint64_t currentFrameId,
      const std::set<LandmarkId>* loopClosureLandmarksToUseExclusively,
      const kinematics::Transformation& T_WS1,
      const AlignedMap<LandmarkId, LandmarkToMatch>& landmarksToMatch,
      size_t numKeypoints,
      const MapPoints& pointMap, size_t im, const MultiFramePtr&  multiFrame,
      std::vector<double>& distances, std::vector<LandmarkId>& lmIds,
      AlignedVector<Eigen::Vector4d>& hps_W, std::vector<size_t>& ctrs) const;

  std::atomic_bool trackingLost_; ///< Is the tracking currently lost?

  /// \brief Hacky: remember which CNNs are running on what frame.
  std::map<StateId,std::vector<std::thread*>> cnnThreads_;

  AlignedVector<Component> components_; ///< Loaded other components.
  std::vector<std::unique_ptr<DBoW>> componentDBows_; ///< Corresponding DBoWs for place recogn.
  std::vector<bool> componentsFixed_; ///< Which ones of the other comp's are to be treated fixed.
};

}  // namespace okvis

#endif // INCLUDE_OKVIS_FRONTEND_HPP_
