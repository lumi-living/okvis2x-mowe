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
 * @file Frontend.cpp
 * @brief Source file for the Frontend class.
 * @author Andreas Forster
 * @author Stefan Leutenegger
 */

#include "okvis/assert_macros.hpp"
#include <opencv2/imgcodecs.hpp>
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <thread>

#include <brisk/brisk.h>

#include <opencv2/imgproc/imgproc.hpp>

#include <glog/logging.h>

// DBoW2
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#include <DBoW2/DBoW2.h>
#pragma GCC diagnostic pop
#include <DBoW2/FBrisk.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <ceres/cost_function.h>
#include <ceres/crs_matrix.h>
#include <ceres/evaluation_callback.h>
#include <ceres/iteration_callback.h>
#include <ceres/loss_function.h>
#include <ceres/manifold.h>
#include <ceres/ordered_groups.h>
#include <ceres/problem.h>
#include <ceres/product_manifold.h>
#include <ceres/sized_cost_function.h>
#include <ceres/solver.h>
#include <ceres/types.h>
#include <ceres/version.h>
#pragma GCC diagnostic pop

#include <okvis/Frontend.hpp>
#include <okvis/DescriptorDistance.hpp>
#include <okvis/LoopClosureGates.hpp>
#include <cstring>
#include <unordered_map>

#include <numeric>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>

#ifdef OKVIS_USE_MOWE_XFEAT
// Mow-e XFeat-on-TensorRT frontend (ADR-0040): replaces BRISK detect+describe;
// LighterGlue replaces brute-force pair matching (stereo / motion stereo).
#include <okvis/xfeat/LighterGlueMatcher.hpp>
#include <okvis/xfeat/XFeatFeatures.hpp>
#include <okvis/xfeat/XFeatFrontend.hpp>
// T-0120: VPR loop closure (DINOv2 embedder, VLAD, brute-force retrieval).
#include <mowe_vpr/embedder.hpp>
#include <mowe_vpr/retrieval.hpp>
#include <mowe_vpr/vlad.hpp>
#endif

// okvis ceres
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/HomogeneousPointParameterBlock.hpp>
#include <okvis/ceres/ReprojectionError.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>
#include <okvis/ceres/HomogeneousPointLocalParameterization.hpp>
#include <okvis/ceres/ImuError.hpp>

// cameras and distortions
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion8.hpp>
#include <okvis/triangulation/stereo_triangulation.hpp>
#include <okvis/cameras/EucmCamera.hpp>

// Kneip RANSAC
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/FrameAbsolutePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/FrameRelativePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/FrameRotationOnlySacProblem.hpp>

#include <okvis/internal/Network.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

static const double kptrad = 0.09;
//cv::Ptr<cv::CLAHE> mClahe;

/// \brief Opaque stuct to use DBoW for loop closure.
class Frontend::DBoW {
 public:
  /// \brief Constructor with Vocabulary directory.
  /// \param dBowVocDir Vocabulary directory.
  DBoW(const std::string& dBowVocDir)
      : vocabulary(dBowVocDir+"/small_voc.yml.gz"),
        // false = do not use direct index.
        database(vocabulary, false, 0)
  {
  }
  /// \brief Constructor with Vocabulary.
  /// \param dBowVoc Vocabulary.
  DBoW(const DBoW2::TemplatedVocabulary<DBoW2::FBrisk::TDescriptor, DBoW2::FBrisk>& dBowVoc)
      : vocabulary(dBowVoc),
      // false = do not use direct index.
      database(vocabulary, false, 0)
  {
  }

  DBoW2::TemplatedVocabulary<DBoW2::FBrisk::TDescriptor, DBoW2::FBrisk> vocabulary; ///< BRISK Voc.
  DBoW2::TemplatedDatabase<DBoW2::FBrisk::TDescriptor, DBoW2::FBrisk> database; ///< BRISK Database.
  std::vector<uint64> poseIds; ///< The multiframe IDs corresponding to the dBow ones.

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// Constructor.
Frontend::Frontend(size_t numCameras, std::string dBowVocDir)
    : isInitialized_(false),
      numCameras_(numCameras),
      briskDetectionOctaves_(0),
      briskDetectionThreshold_(40.0),
      briskDetectionAbsoluteThreshold_(200.0),
      briskDetectionMaximumKeypoints_(450),
      briskDescriptionRotationInvariance_(true),
      briskDescriptionScaleInvariance_(false),
      briskMatchingThreshold_(60.0),
      keyframeInsertionOverlapThreshold_(0.55f),
      dBowVocDir_(dBowVocDir)
{
  // create mutexes for feature detectors and descriptor extractors
  for (size_t i = 0; i < numCameras_; ++i) {
    featureDetectorMutexes_.push_back(std::unique_ptr<std::mutex>(new std::mutex()));
  }
  trackingLost_ = false;
  initialiseBriskFeatureDetectors();

#ifdef OKVIS_USE_NN
  // Deserialize the ScriptModule from a file using torch::jit::load().
  networks_.resize(numCameras);
  for (size_t i = 0; i < numCameras_; ++i) {
#ifdef OKVIS_USE_GPU
#ifdef OKVIS_USE_MPS
    networks_[i].reset(new Network(torch::jit::load(dBowVocDir+"/fast-scnn.pt", torch::kCPU)));
    networks_[i]->to(torch::kMPS);
#else
    networks_[i].reset(new Network(torch::jit::load(dBowVocDir+"/fast-scnn.pt", torch::kCUDA)));
    networks_[i]->to(torch::kCUDA);
#endif
#else
    networks_[i].reset(new Network(torch::jit::load(dBowVocDir+"/fast-scnn.pt", torch::kCPU)));
    networks_[i]->to(torch::kCPU);
#endif
  }

#endif
}

// ---- Mow-e XFeat frontend (ADR-0040) ---------------------------------------

/// \brief Owns the per-camera XFeat TensorRT engines and the LighterGlue
///        matcher (PIMPL keeps TensorRT types out of Frontend.hpp; only
///        populated in USE_MOWE_XFEAT builds).
struct Frontend::XFeatRuntime {
#ifdef OKVIS_USE_MOWE_XFEAT
  /// Stereo rig (2 cameras): ONE batch-2 engine, run once per MultiFrame on
  /// the (L,R) pair under pairMutex (T-0113; the second camera's detection
  /// thread finds the pair done and returns). Otherwise one engine per camera:
  /// detectAndDescribe runs per-camera in parallel and a TensorRT execution
  /// context + stream pair is not thread-safe.
  std::vector<std::unique_ptr<xfeat::XFeatFrontend>> engines;
  std::mutex pairMutex;
  bool pairDone = false;      ///< pairDoneStamp valid
  okvis::Time pairDoneStamp;  ///< MultiFrame timestamp the pair was last run on
  /// Pair matcher for stereo / motion stereo (stage B/C); null → cosine NN.
  std::unique_ptr<xfeat::LighterGlueMatcher> lighterGlue;
  /// One matcher context — serialise match() calls (defensive; the matching
  /// stages run sequentially in dataAssociationAndInitialization today).
  std::mutex lighterGlueMutex;
#endif
};

void Frontend::setXFeatParameters(const XFeatParameters& xfeat) {
  xfeatParams_ = xfeat;
  floatDescriptors_ = xfeat.use;  // XFeat -> 64-D float rows, cosine metric
  if (!xfeat.use) {
    xfeatRuntime_.reset();
    return;
  }
#ifdef OKVIS_USE_MOWE_XFEAT
  std::unique_ptr<XFeatRuntime> runtime(new XFeatRuntime());
  const size_t numEngines = numCameras_ == 2 ? 1 : numCameras_;
  for (size_t i = 0; i < numEngines; ++i) {
    xfeat::XFeatConfig cfg;
    cfg.engine_path = xfeat.engine;
    cfg.score_threshold = float(xfeat.score_threshold);
    runtime->engines.emplace_back(new xfeat::XFeatFrontend(cfg));
    OKVIS_ASSERT_TRUE(Exception, runtime->engines.back()->engine_loaded(),
                      "XFeat engine failed to load (TensorRT build + .plan "
                      "required): " << xfeat.engine)
  }
  if (!xfeat.lighterglue_engine.empty()) {
    xfeat::LighterGlueConfig lgCfg;
    lgCfg.engine_path = xfeat.lighterglue_engine;
    lgCfg.min_score = float(xfeat.match_score_min);
    runtime->lighterGlue.reset(new xfeat::LighterGlueMatcher(lgCfg));
    OKVIS_ASSERT_TRUE(Exception, runtime->lighterGlue->loaded(),
                      "LighterGlue engine failed to load: "
                          << xfeat.lighterglue_engine)
  }
  std::uint32_t w = 0, h = 0;
  runtime->engines.front()->input_dims(w, h);
  xfeatRuntime_ = std::move(runtime);
  LOG(INFO) << "XFeat frontend enabled: " << xfeat.engine << " (" << w << "x"
            << h << "); matching_threshold " << briskMatchingThreshold_
            << " interpreted as cosine distance";
  if (xfeatRuntime_->lighterGlue) {
    LOG(INFO) << "LighterGlue pair matcher enabled: "
              << xfeat.lighterglue_engine << " (capacity "
              << xfeatRuntime_->lighterGlue->capacity() << ", min score "
              << xfeat.match_score_min << ")";
  } else {
    LOG(INFO) << "LighterGlue not configured — cosine NN pair matching";
  }
#else
  OKVIS_THROW(Exception,
              "frontend_parameters: xfeat: use=true, but okvis was built "
              "without USE_MOWE_XFEAT")
#endif
}

bool Frontend::usingXFeat() const {
  return xfeatParams_.use && xfeatRuntime_ != nullptr;
}

// -----------------------------------------------------------------------------
// VPR loop closure (Mow-e T-0120, mowe-nav-kb 06 §adapter, ADR-0042): the
// per-keyframe "descriptor -> candidate keyframe ids" query is DINOv2 ViT-S/14
// + VLAD (onboard/localization/mowe_vpr) instead of DBoW2; the candidates go
// through prior gate -> LighterGlue -> GP3P RANSAC -> temporal consistency and
// then into the UNCHANGED ViSlamBackend::attemptLoopClosure path.
struct Frontend::VprLoop {
#ifdef OKVIS_USE_MOWE_XFEAT
  mowe_vpr::TrtEmbedder embedder;
  mowe_vpr::Vocabulary vocabulary;
  /// In-session database: Database id == index into entries.
  struct Entry {
    uint64_t stateId;
    double cumDist;  ///< path length [m] along keyframes when this one was added
  };
  mowe_vpr::Database db;
  std::vector<Entry> entries;
  std::unordered_map<uint64_t, int> indexOfState;
  double cumDist = 0.0;
  bool haveLast = false;
  Eigen::Vector3d r_last = Eigen::Vector3d::Zero();
  /// Foreign map (--preload-map): Database id == index into foreignFrames.
  mowe_vpr::Database foreignDb;
  std::vector<std::shared_ptr<const MultiFrame>> foreignFrames;
  std::vector<uint64_t> foreignIds;
  loopclosure::TemporalConsistency temporal{2, 5};
  loopclosure::TemporalConsistency temporalForeign{2, 5};
  /// Current keyframe descriptor (computed once per keyframe, queried then added).
  Eigen::VectorXf currentDesc;
  uint64_t currentDescFrameId = 0;
  bool haveCurrentDesc = false;
  /// Serialise verifyRecognisedPlace timing etc. — single processing thread today.
#endif
};

void Frontend::setVprLoopParameters(const VprLoopParameters& vpr) {
  vprParams_ = vpr;
  vprLoop_.reset();
  if (vpr.engine.empty()) {
    return;
  }
#ifdef OKVIS_USE_MOWE_XFEAT
  OKVIS_ASSERT_TRUE(Exception, floatDescriptors_,
                    "frontend_parameters: vpr: needs the XFeat (float descriptor) frontend")
  std::unique_ptr<VprLoop> loop(new VprLoop());
  OKVIS_ASSERT_TRUE(Exception, loop->embedder.load(vpr.engine),
                    "DINOv2 VPR engine failed to load: " << vpr.engine)
  OKVIS_ASSERT_TRUE(Exception, loop->vocabulary.load(vpr.vocabulary),
                    "VPR vocabulary failed to load: " << vpr.vocabulary)
  OKVIS_ASSERT_TRUE(Exception, loop->vocabulary.dim() == loop->embedder.dim(),
                    "VPR vocabulary token dim " << loop->vocabulary.dim()
                        << " != engine dim " << loop->embedder.dim())
  loop->temporal = loopclosure::TemporalConsistency(vpr.consecutive_required, vpr.consecutive_max_gap);
  loop->temporalForeign = loopclosure::TemporalConsistency(vpr.consecutive_required, vpr.consecutive_max_gap);
  vprLoop_ = std::move(loop);
  LOG(INFO) << "VPR loop closure enabled: " << vpr.engine << " + " << vpr.vocabulary
            << " (K=" << vprLoop_->vocabulary.k() << ", desc dim "
            << vprLoop_->vocabulary.desc_dim() << "); top_k " << vpr.top_k
            << ", score_min " << vpr.score_min << ", prior gate " << vpr.prior_gate_sigma
            << " sigma, min_inliers " << vpr.min_inliers << ", reproj " << vpr.reproj_px
            << " px, min_inlier_ratio " << vpr.min_inlier_ratio << ", consecutive "
            << vpr.consecutive_required;
#else
  OKVIS_THROW(Exception,
              "frontend_parameters: vpr: engine set, but okvis was built without USE_MOWE_XFEAT")
#endif
}

bool Frontend::usingVprLoopClosure() const {
  return vprLoop_ != nullptr;
}

bool Frontend::vprEmbedCurrent(const MultiFrame& frame) {
#ifdef OKVIS_USE_MOWE_XFEAT
  vprLoop_->haveCurrentDesc = false;
  const cv::Mat& image = frame.image(0);
  if (image.empty()) {
    LOG(WARNING) << "VPR: keyframe " << frame.id() << " has no left image; skipped";
    return false;
  }
  const auto t0 = std::chrono::steady_clock::now();
  Eigen::VectorXf cls;
  Eigen::MatrixXf patch;
  if (!vprLoop_->embedder.embed(image, cls, patch)) {
    LOG(WARNING) << "VPR: embedding failed on keyframe " << frame.id();
    return false;
  }
  patch.rowwise().normalize();  // vpr_eval --norm-tokens 1 (the vocabulary was fitted so)
  vprLoop_->currentDesc = mowe_vpr::describe(patch, vprLoop_->vocabulary);
  vprLoop_->currentDescFrameId = frame.id();
  vprLoop_->haveCurrentDesc = true;
  loopStats_.embedMs.push_back(std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - t0).count());
  return true;
#else
  (void)frame;
  return false;
#endif
}

void Frontend::vprQuery(const Estimator& estimator, const okvis::ViParameters& params,
                        std::shared_ptr<okvis::MultiFrame> framesInOut,
                        std::vector<std::pair<StateId, double>>& stateIds) {
#ifdef OKVIS_USE_MOWE_XFEAT
  VprLoop& L = *vprLoop_;
  ++loopStats_.queries;
  // in-session candidates, score-descending (Database::query sorts)
  for (const mowe_vpr::Candidate& c : L.db.query(L.currentDesc, vprParams_.top_k)) {
    if (c.score < float(vprParams_.score_min)) {
      continue;
    }
    ++loopStats_.candidates;
    stateIds.push_back(std::make_pair(StateId(L.entries.at(size_t(c.id)).stateId), double(c.score)));
  }
  // foreign map: verified and counted, never handed to the estimator (T-0121 owns
  // the relocalisation against a saved map).
  if (L.foreignDb.size() > 0) {
    bool verified = false;
    for (const mowe_vpr::Candidate& c : L.foreignDb.query(L.currentDesc, vprParams_.top_k)) {
      if (c.score < float(vprParams_.score_min)) {
        continue;
      }
      ++loopStats_.foreignCandidates;
      kinematics::Transformation T_Sold_Snew;
      Eigen::Matrix<double, 6, 6> H;
      const auto t0 = std::chrono::steady_clock::now();
      const bool ok = verifyRecognisedPlace(estimator, params, framesInOut,
                                            L.foreignFrames.at(size_t(c.id)), T_Sold_Snew, H,
                                            vprParams_.min_inliers);
      loopStats_.verificationMs.push_back(std::chrono::duration<double, std::milli>(
                                              std::chrono::steady_clock::now() - t0).count());
      if (!ok) {
        ++loopStats_.foreignRejectedByGeometry;
        continue;
      }
      verified = true;
      if (!L.temporalForeign.push(c.id)) {
        ++loopStats_.foreignRejectedByTemporal;
        break;
      }
      ++loopStats_.loopsAcceptedAgainstForeignMap;
      loopStats_.foreignLoops.push_back(
          {framesInOut->id(), uint64_t(framesInOut->timestamp().toNSec()),
           L.foreignIds.at(size_t(c.id)),
           uint64_t(L.foreignFrames.at(size_t(c.id))->timestamp().toNSec()),
           double(c.score), T_Sold_Snew.r().norm()});
      LOG(WARNING) << "FOREIGN-MAP LOOP: frame " << framesInOut->id() << " t_ns "
                   << framesInOut->timestamp().toNSec() << " -> foreign keyframe "
                   << L.foreignIds.at(size_t(c.id)) << " t_ns "
                   << L.foreignFrames.at(size_t(c.id))->timestamp().toNSec() << " (score " << c.score
                   << ", t_Sold_Snew " << T_Sold_Snew.r().transpose()
                   << ") — counted only, not fed to the graph (T-0121)";
      break;
    }
    if (!verified) {
      L.temporalForeign.miss();
    }
  }
#else
  (void)estimator; (void)params; (void)framesInOut; (void)stateIds;
#endif
}

void Frontend::vprAddCurrent(const Estimator& estimator, const MultiFrame& frame) {
#ifdef OKVIS_USE_MOWE_XFEAT
  VprLoop& L = *vprLoop_;
  if (!L.haveCurrentDesc || L.currentDescFrameId != frame.id()) {
    return;
  }
  const Eigen::Vector3d r_WS = estimator.pose(StateId(frame.id())).r();
  if (L.haveLast) {
    L.cumDist += (r_WS - L.r_last).norm();
  }
  L.r_last = r_WS;
  L.haveLast = true;
  L.indexOfState[frame.id()] = int(L.entries.size());
  L.db.add(int(L.entries.size()), L.currentDesc);
  L.entries.push_back({frame.id(), L.cumDist});
  L.haveCurrentDesc = false;
#else
  (void)estimator; (void)frame;
#endif
}

Frontend::LoopStats Frontend::loopStats() const {
  std::lock_guard<std::mutex> lock(statsMutex_);
  return loopStats_;
}

bool Frontend::writeLoopStatsJson(const std::string& path) const {
  const LoopStats s = loopStats();
  auto summary = [](std::vector<double> v, double& mean, double& p99, double& mx) {
    std::sort(v.begin(), v.end());
    mean = v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / double(v.size());
    p99 = v.empty() ? 0.0 : v[std::min(v.size() - 1, size_t(0.99 * double(v.size())))];
    mx = v.empty() ? 0.0 : v.back();
  };
  double vMean, vP99, vMax, eMean, eP99, eMax, mMean, mP99, mMax;
  summary(s.verificationMs, vMean, vP99, vMax);
  summary(s.embedMs, eMean, eP99, eMax);
  summary(s.priorMahalanobis, mMean, mP99, mMax);
  size_t dbSize = 0, foreignSize = 0;
#ifdef OKVIS_USE_MOWE_XFEAT
  if (vprLoop_) {
    dbSize = vprLoop_->entries.size();
    foreignSize = vprLoop_->foreignFrames.size();
  }
#endif
  std::ofstream f(path);
  if (!f.good()) {
    LOG(ERROR) << "cannot write loop stats to " << path;
    return false;
  }
  f << std::setprecision(6) << std::fixed << "{\n"
    << "  \"vpr_enabled\": " << (vprLoop_ ? 1 : 0) << ",\n"
    << "  \"engine\": \"" << vprParams_.engine << "\",\n"
    << "  \"vocabulary\": \"" << vprParams_.vocabulary << "\",\n"
    << "  \"top_k\": " << vprParams_.top_k << ",\n"
    << "  \"score_min\": " << vprParams_.score_min << ",\n"
    << "  \"prior_gate_sigma\": " << vprParams_.prior_gate_sigma << ",\n"
    << "  \"prior_sigma_pos_m\": " << vprParams_.prior_sigma_pos_m << ",\n"
    << "  \"prior_drift_frac\": " << vprParams_.prior_drift_frac << ",\n"
    << "  \"prior_sigma_rot_deg\": " << vprParams_.prior_sigma_rot_deg << ",\n"
    << "  \"min_inliers\": " << vprParams_.min_inliers << ",\n"
    << "  \"reproj_px\": " << vprParams_.reproj_px << ",\n"
    << "  \"min_inlier_ratio\": " << vprParams_.min_inlier_ratio << ",\n"
    << "  \"consecutive_required\": " << vprParams_.consecutive_required << ",\n"
    << "  \"consecutive_max_gap\": " << vprParams_.consecutive_max_gap << ",\n"
    << "  \"keyframes_in_db\": " << dbSize << ",\n"
    << "  \"foreign_keyframes\": " << foreignSize << ",\n"
    << "  \"vpr_queries\": " << s.queries << ",\n"
    << "  \"candidates\": " << s.candidates << ",\n"
    << "  \"candidates_skipped_state\": " << s.candidatesSkippedState << ",\n"
    << "  \"candidates_rejected_by_prior_gate\": " << s.rejectedByPriorGate << ",\n"
    << "  \"candidates_rejected_by_geometry\": " << s.rejectedByGeometry << ",\n"
    << "  \"candidates_rejected_by_temporal\": " << s.rejectedByTemporal << ",\n"
    << "  \"candidates_rejected_by_estimator\": " << s.rejectedByEstimator << ",\n"
    << "  \"loops_accepted\": " << s.loopsAccepted << ",\n"
    << "  \"foreign_candidates\": " << s.foreignCandidates << ",\n"
    << "  \"foreign_rejected_by_geometry\": " << s.foreignRejectedByGeometry << ",\n"
    << "  \"foreign_rejected_by_temporal\": " << s.foreignRejectedByTemporal << ",\n"
    << "  \"loops_accepted_against_foreign_map\": " << s.loopsAcceptedAgainstForeignMap << ",\n"
    << "  \"verifications\": " << s.verificationMs.size() << ",\n"
    << "  \"mean_verification_ms\": " << vMean << ",\n"
    << "  \"p99_verification_ms\": " << vP99 << ",\n"
    << "  \"max_verification_ms\": " << vMax << ",\n"
    << "  \"embeds\": " << s.embedMs.size() << ",\n"
    << "  \"mean_embed_ms\": " << eMean << ",\n"
    << "  \"p99_embed_ms\": " << eP99 << ",\n"
    << "  \"prior_gated_candidates\": " << s.priorMahalanobis.size() << ",\n"
    << "  \"mean_prior_mahalanobis\": " << mMean << ",\n"
    << "  \"max_prior_mahalanobis\": " << mMax << ",\n"
    << "  \"foreign_loops\": [";
  for (size_t i = 0; i < s.foreignLoops.size(); ++i) {
    const auto& l = s.foreignLoops[i];
    f << (i ? ",\n    " : "\n    ") << "{\"frame\": " << l.frameId << ", \"t_ns\": " << l.queryNs
      << ", \"keyframe\": " << l.keyframeId << ", \"keyframe_t_ns\": " << l.keyframeNs
      << ", \"score\": " << l.score << ", \"t_verified_m\": " << l.tNormM << "}";
  }
  f << (s.foreignLoops.empty() ? "]\n" : "\n  ]\n") << "}\n";
  LOG(INFO) << "loop stats written to " << path;
  return true;
}

// keyframes.bin (T-0120, seed of the T-0121 map format), little-endian:
//   "MOWEKF01" u32 numCameras u32 vprDim u32 numKeyframes
//   per keyframe: u64 id, i64 t_ns, f64 r_WS[3], f64 q_WS[4] (x y z w), f32 vpr[vprDim],
//     per camera: u32 K, per keypoint: f32 x y size response, f32 desc[64], u64 lmId,
//                 u8 initialised, f64 hp_S[4]   (landmark in THIS keyframe's S frame)
namespace {
constexpr char kKeyframesMagic[8] = {'M', 'O', 'W', 'E', 'K', 'F', '0', '1'};
template <typename T> bool wr(FILE* f, const T& v) { return std::fwrite(&v, sizeof(T), 1, f) == 1; }
template <typename T> bool rd(FILE* f, T& v) { return std::fread(&v, sizeof(T), 1, f) == 1; }
}  // namespace

bool Frontend::saveKeyframes(const Estimator& estimator, const std::string& path) const {
#ifdef OKVIS_USE_MOWE_XFEAT
  if (!vprLoop_) {
    return false;
  }
  const VprLoop& L = *vprLoop_;
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) {
    LOG(ERROR) << "cannot write " << path;
    return false;
  }
  bool ok = std::fwrite(kKeyframesMagic, 1, 8, f) == 8;
  const uint32_t vprDim = uint32_t(L.db.dim());
  // count saveable keyframes first (multi-frame still held by the estimator)
  std::vector<const VprLoop::Entry*> saveable;
  for (const auto& e : L.entries) {
    if (estimator.multiFrame(StateId(e.stateId))) {
      saveable.push_back(&e);
    }
  }
  ok = ok && wr(f, uint32_t(numCameras_)) && wr(f, vprDim) && wr(f, uint32_t(saveable.size()));
  size_t numLandmarks = 0;
  for (size_t n = 0; n < saveable.size() && ok; ++n) {
    const VprLoop::Entry& e = *saveable[n];
    const int dbIndex = L.indexOfState.at(e.stateId);
    const MultiFramePtr mf = estimator.multiFrame(StateId(e.stateId));
    const kinematics::Transformation T_WS = estimator.pose(StateId(e.stateId));
    const kinematics::Transformation T_SW = T_WS.inverse();
    ok = ok && wr(f, uint64_t(e.stateId)) && wr(f, int64_t(mf->timestamp().toNSec()));
    for (int i = 0; i < 3 && ok; ++i) ok = wr(f, double(T_WS.r()[i]));
    const Eigen::Vector4d q = T_WS.q().coeffs();  // x y z w
    for (int i = 0; i < 4 && ok; ++i) ok = wr(f, double(q[i]));
    // the descriptor: Database rows are the stored (unit) descriptors
    const Eigen::VectorXf desc = L.db.row(dbIndex);
    ok = ok && std::fwrite(desc.data(), sizeof(float), size_t(vprDim), f) == size_t(vprDim);
    for (size_t im = 0; im < numCameras_ && ok; ++im) {
      const size_t K = mf->numKeypoints(im);
      ok = wr(f, uint32_t(K));
      for (size_t k = 0; k < K && ok; ++k) {
        cv::KeyPoint kp;
        mf->getCvKeypoint(im, k, kp);
        ok = wr(f, float(kp.pt.x)) && wr(f, float(kp.pt.y)) && wr(f, float(kp.size)) && wr(f, float(kp.response));
        ok = ok && std::fwrite(mf->keypointDescriptor(im, k), sizeof(float), size_t(kFloatDescriptorDim), f) == size_t(kFloatDescriptorDim);
        const uint64_t lmId = mf->landmarkId(im, k);
        Eigen::Vector4d hp_S(0, 0, 0, 0);
        bool initialised = false;
        mf->getLandmark(im, k, hp_S, initialised);
        if (!initialised && lmId != 0 && estimator.isLandmarkAdded(LandmarkId(lmId))) {
          // still a live landmark in the real-time graph (not yet a pose-graph frame)
          MapPoint2 mp;
          if (estimator.getLandmark(LandmarkId(lmId), mp)) {
            hp_S = T_SW * Eigen::Vector4d(mp.point);
            initialised = mp.isInitialised;
          }
        }
        if (initialised) ++numLandmarks;
        ok = ok && wr(f, lmId) && wr(f, uint8_t(initialised ? 1 : 0));
        for (int i = 0; i < 4 && ok; ++i) ok = wr(f, double(hp_S[i]));
      }
    }
  }
  ok = (std::fclose(f) == 0) && ok;
  LOG(INFO) << "keyframes written to " << path << ": " << saveable.size() << " of "
            << L.entries.size() << " keyframes, " << numLandmarks << " initialised landmarks";
  return ok;
#else
  (void)estimator; (void)path;
  return false;
#endif
}

bool Frontend::loadForeignKeyframes(const std::string& path,
                                    const cameras::NCameraSystem& cameraSystem) {
#ifdef OKVIS_USE_MOWE_XFEAT
  OKVIS_ASSERT_TRUE(Exception, vprLoop_, "--preload-map needs frontend_parameters.vpr")
  VprLoop& L = *vprLoop_;
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    LOG(ERROR) << "cannot read " << path;
    return false;
  }
  char magic[8];
  uint32_t numCameras = 0, vprDim = 0, numKeyframes = 0;
  bool ok = std::fread(magic, 1, 8, f) == 8 && std::memcmp(magic, kKeyframesMagic, 8) == 0 &&
            rd(f, numCameras) && rd(f, vprDim) && rd(f, numKeyframes);
  OKVIS_ASSERT_TRUE(Exception, ok, "not a MOWEKF01 file: " << path)
  OKVIS_ASSERT_TRUE(Exception, numCameras == cameraSystem.numCameras(),
                    "keyframes.bin has " << numCameras << " cameras, config has " << cameraSystem.numCameras())
  OKVIS_ASSERT_TRUE(Exception, int(vprDim) == L.vocabulary.desc_dim(),
                    "keyframes.bin VPR dim " << vprDim << " != vocabulary " << L.vocabulary.desc_dim())
  size_t numLandmarks = 0;
  for (uint32_t n = 0; n < numKeyframes && ok; ++n) {
    uint64_t id = 0;
    int64_t t_ns = 0;
    double r[3], q[4];
    ok = rd(f, id) && rd(f, t_ns);
    for (int i = 0; i < 3 && ok; ++i) ok = rd(f, r[i]);
    for (int i = 0; i < 4 && ok; ++i) ok = rd(f, q[i]);
    Eigen::VectorXf desc(vprDim);
    ok = ok && std::fread(desc.data(), sizeof(float), vprDim, f) == vprDim;
    std::shared_ptr<MultiFrame> mf(new MultiFrame(cameraSystem, okvis::Time().fromNSec(uint64_t(t_ns)), id));
    for (uint32_t im = 0; im < numCameras && ok; ++im) {
      uint32_t K = 0;
      ok = rd(f, K);
      std::vector<cv::KeyPoint> kps(K);
      cv::Mat descriptors(int(K), kFloatDescriptorDim, CV_32FC1);
      std::vector<uint64_t> lmIds(K);
      std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> hps(K);
      std::vector<bool> init(K);
      for (uint32_t k = 0; k < K && ok; ++k) {
        float x, y, size, response;
        ok = rd(f, x) && rd(f, y) && rd(f, size) && rd(f, response);
        kps[k] = cv::KeyPoint(x, y, size, -1.f, response);
        ok = ok && std::fread(descriptors.ptr<float>(int(k)), sizeof(float), size_t(kFloatDescriptorDim), f) == size_t(kFloatDescriptorDim);
        uint8_t initialised = 0;
        ok = ok && rd(f, lmIds[k]) && rd(f, initialised);
        for (int i = 0; i < 4 && ok; ++i) ok = rd(f, hps[k][i]);
        init[k] = initialised != 0;
      }
      if (!ok) break;
      mf->resetKeypoints(im, kps);
      mf->resetDescriptors(im, descriptors);
      for (uint32_t k = 0; k < K; ++k) {
        mf->setLandmarkId(im, k, lmIds[k]);
        mf->setLandmark(im, k, hps[k], init[k]);
        if (init[k]) ++numLandmarks;
      }
    }
    if (!ok) break;
    L.foreignDb.add(int(L.foreignFrames.size()), desc);
    L.foreignFrames.push_back(mf);
    L.foreignIds.push_back(id);
  }
  std::fclose(f);
  OKVIS_ASSERT_TRUE(Exception, ok, "truncated keyframes.bin: " << path)
  LOG(INFO) << "foreign map loaded from " << path << ": " << L.foreignFrames.size()
            << " keyframes, " << numLandmarks << " landmarks (retrieval + verification only)";
  return true;
#else
  (void)path; (void)cameraSystem;
  OKVIS_THROW(Exception, "--preload-map needs a USE_MOWE_XFEAT build")
  return false;
#endif
}

// Descriptor metric seam (okvis/DescriptorDistance.hpp, T-0112): the matching
// loops below work on raw rows and dispatch here; matching_threshold is
// interpreted on the active scale (Hamming bits, or cosine distance, ADR-0040).
static constexpr int kXFeatDescriptorDim = kFloatDescriptorDim;

size_t Frontend::descriptorBytes() const {
  return DescriptorMetric{floatDescriptors_}.bytes();
}

double Frontend::descriptorDist(const unsigned char* a,
                                const unsigned char* b) const {
  return DescriptorMetric{floatDescriptors_}(a, b);
}

Frontend::DBoW& Frontend::dBow() {
  OKVIS_ASSERT_TRUE(Exception, !floatDescriptors_,
                    "DBoW2 requested on the float-descriptor path: the "
                    "vocabulary is BRISK-trained (ADR-0040, T-0112)")
  if (!dBow_) {
    dBow_.reset(new DBoW(dBowVocDir_));
  }
  return *dBow_;
}

#ifdef OKVIS_USE_MOWE_XFEAT
/// \brief Store one camera's XFeat features (strongest maxKeypoints) into the
///        MultiFrame: keypoints, CV_32F [n x 64] descriptors, back-projections.
static void injectXFeatFeatures(const xfeat::StreamFeatures& features,
                                size_t cameraIndex, size_t maxKeypoints,
                                float keypointSize, okvis::MultiFrame& frameOut) {
  // Keep only the strongest max_num_keypoints, like the BRISK detector does.
  std::vector<size_t> order(features.size());
  std::iota(order.begin(), order.end(), size_t(0));
  const size_t numKeypoints = std::min(features.size(), maxKeypoints);
  if (numKeypoints < features.size()) {
    std::partial_sort(order.begin(), order.begin() + numKeypoints, order.end(),
                      [&features](size_t a, size_t b) {
                        return features.scores[a] > features.scores[b];
                      });
  }
  std::vector<cv::KeyPoint> keypoints;
  keypoints.reserve(numKeypoints);
  cv::Mat descriptors(int(numKeypoints), kXFeatDescriptorDim, CV_32FC1);
  for (size_t n = 0; n < numKeypoints; ++n) {
    const size_t i = order[n];
    // XFeat has no scale; keypoint_size sets the observation uncertainty
    // (sigma = size/focalLength * 0.125 in the matching stages).
    keypoints.emplace_back(features.keypoints_px[i].u,
                           features.keypoints_px[i].v, keypointSize, -1.f,
                           features.scores[i]);
    std::memcpy(descriptors.ptr<float>(int(n)),
                &features.descriptors[i * xfeat::kDescriptorDim],
                sizeof(float) * xfeat::kDescriptorDim);
  }
  frameOut.resetKeypoints(cameraIndex, keypoints);
  frameOut.resetDescriptors(cameraIndex, descriptors);
  frameOut.computeBackProjections(cameraIndex);
}
#endif

bool Frontend::detectAndDescribeXFeat(
    size_t cameraIndex, std::shared_ptr<okvis::MultiFrame> frameOut) {
#ifdef OKVIS_USE_MOWE_XFEAT
  const float keypointSize = float(xfeatParams_.keypoint_size);
  const auto t0 = std::chrono::steady_clock::now();
  auto recordMs = [&](uint64_t keypoints, uint64_t frames) {
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();
    std::lock_guard<std::mutex> lock(statsMutex_);
    stats_.frontendMs.push_back(ms);
    stats_.keypoints += keypoints;
    stats_.frames += frames;
  };

  // Stereo pair: one batch-2 enqueue per MultiFrame (T-0113, KB 03 §batching).
  if (numCameras_ == 2 && frameOut->numFrames() == 2 &&
      !frameOut->image(0).empty() && !frameOut->image(1).empty()) {
    std::lock_guard<std::mutex> lock(xfeatRuntime_->pairMutex);
    if (xfeatRuntime_->pairDone &&
        xfeatRuntime_->pairDoneStamp == frameOut->timestamp()) {
      return true;  // the other camera's detection thread already ran the pair
    }
    const cv::Mat left = frameOut->image(0), right = frameOut->image(1);
    OKVIS_ASSERT_TRUE(Exception,
                      left.type() == CV_8UC1 && right.type() == CV_8UC1 &&
                          left.size() == right.size(),
                      "XFeat pair path expects two equal-size mono8 images")
    xfeat::FrameFeatures pair = xfeatRuntime_->engines.front()->extract_pair(
        left.data, std::uint32_t(left.step), right.data,
        std::uint32_t(right.step), std::uint32_t(left.cols),
        std::uint32_t(left.rows));
    for (size_t im = 0; im < 2; ++im) {
      injectXFeatFeatures(pair.streams.at(im), im,
                          briskDetectionMaximumKeypoints_, keypointSize,
                          *frameOut);
    }
    xfeatRuntime_->pairDone = true;
    xfeatRuntime_->pairDoneStamp = frameOut->timestamp();
    recordMs(frameOut->numKeypoints(0) + frameOut->numKeypoints(1), 2);
    return true;
  }

  // Single image (mono rig, or a camera without a partner image).
  xfeat::XFeatFrontend& engine = *xfeatRuntime_->engines.at(
      std::min(cameraIndex, xfeatRuntime_->engines.size() - 1));
  const cv::Mat image = frameOut->image(cameraIndex);
  OKVIS_ASSERT_TRUE(Exception, image.type() == CV_8UC1,
                    "XFeat frontend expects mono8 images")
  xfeat::StreamFeatures features =
      engine.extract_image(image.data, std::uint32_t(image.step),
                           std::uint32_t(image.cols), std::uint32_t(image.rows));
  injectXFeatFeatures(features, cameraIndex, briskDetectionMaximumKeypoints_,
                      keypointSize, *frameOut);
  recordMs(frameOut->numKeypoints(cameraIndex), 1);
  return true;
#else
  (void)cameraIndex;
  (void)frameOut;
  OKVIS_THROW(Exception, "not built with USE_MOWE_XFEAT")
  return false;
#endif
}

bool Frontend::lighterGluePairProposals(const okvis::MultiFrame& frameA,
                                        size_t imA,
                                        const okvis::MultiFrame& frameB,
                                        size_t imB,
                                        std::vector<int>& matchBForA) {
#ifdef OKVIS_USE_MOWE_XFEAT
  if (!usingXFeat() || !xfeatRuntime_->lighterGlue) {
    return false;
  }
  const size_t nA = frameA.numKeypoints(imA);
  const size_t nB = frameB.numKeypoints(imB);
  matchBForA.assign(nA, -1);
  if (nA == 0 || nB == 0) {
    return true;  // valid LighterGlue result: no possible matches
  }

  // Gather pixel coords; descriptors are already a contiguous [n x 64] float
  // block (the CV_32F Mat built by detectAndDescribeXFeat).
  // Scores (cv::KeyPoint::response = XFeat score) pick the top-K the matcher
  // stages and drive its validity mask (score > 0). // T-0114
  std::vector<float> kptsA(nA * 2), kptsB(nB * 2), scoresA(nA), scoresB(nB);
  cv::KeyPoint kp;
  for (size_t k = 0; k < nA; ++k) {
    frameA.getCvKeypoint(imA, k, kp);
    kptsA[k * 2] = kp.pt.x;
    kptsA[k * 2 + 1] = kp.pt.y;
    scoresA[k] = kp.response;
  }
  for (size_t k = 0; k < nB; ++k) {
    frameB.getCvKeypoint(imB, k, kp);
    kptsB[k * 2] = kp.pt.x;
    kptsB[k * 2 + 1] = kp.pt.y;
    scoresB[k] = kp.response;
  }
  const float* descA =
      reinterpret_cast<const float*>(frameA.keypointDescriptor(imA, 0));
  const float* descB =
      reinterpret_cast<const float*>(frameB.keypointDescriptor(imB, 0));

  std::lock_guard<std::mutex> lock(xfeatRuntime_->lighterGlueMutex);
  const xfeat::PairMatches matches = xfeatRuntime_->lighterGlue->match(
      kptsA.data(), descA, scoresA.data(), nA,
      std::uint32_t(frameA.geometry(imA)->imageWidth()),
      std::uint32_t(frameA.geometry(imA)->imageHeight()), kptsB.data(), descB,
      scoresB.data(), nB, std::uint32_t(frameB.geometry(imB)->imageWidth()),
      std::uint32_t(frameB.geometry(imB)->imageHeight()));
  for (size_t m = 0; m < matches.size(); ++m) {
    matchBForA[matches.indices[m].first] = int(matches.indices[m].second);
  }
  return true;
#else
  (void)frameA;
  (void)imA;
  (void)frameB;
  (void)imB;
  (void)matchBForA;
  return false;
#endif
}

// -----------------------------------------------------------------------------

Frontend::~Frontend() {
  endCnnThreads();
}

bool Frontend::loadComponent(std::string filename,
                             const ImuParameters &imuParameters,
                             const cameras::NCameraSystem &nCameraSystem,
                             bool componentFixed)
{
  OKVIS_ASSERT_TRUE(Exception, !floatDescriptors(),
                    "loadComponent: saved components store BRISK descriptors "
                    "and a BRISK DBoW vocabulary; not usable with the XFeat "
                    "frontend (ADR-0040)")

  // create component
  components_.emplace_back(Component(imuParameters, nCameraSystem));
  componentsFixed_.push_back(componentFixed);

  // load it
  if (!components_.back().load(filename)) {
    return false;
  }

  // create component DBoW
  componentDBows_.emplace_back(std::unique_ptr<DBoW>(new DBoW(dBow().vocabulary)));

  // fill component DBoW
  for (const auto &multiFrame : components_.back().multiFrames_) {
    // first get features
    std::vector<std::vector<uchar>> features(multiFrame.second->numKeypoints());
    int offset = 0;
    for (size_t im = 0; im < numCameras_; ++im) {
      for (size_t k = 0; k < multiFrame.second->numKeypoints(im); ++k) {
        features.at(k + offset).resize(48); // TODO: get 48 from feature
        memcpy(features.at(k + offset).data(),
               multiFrame.second->keypointDescriptor(im, k),
               48 * sizeof(uchar));

      }
      offset += multiFrame.second->numKeypoints(im);
    }
    // ... and add:
    componentDBows_.back()->database.add(features);
    componentDBows_.back()->poseIds.push_back(multiFrame.second->id());
  }

  return true;
}

// Detection and descriptor extraction on a per image basis.
bool Frontend::detectAndDescribe(size_t cameraIndex, std::shared_ptr<okvis::MultiFrame> frameOut,
                                 const okvis::kinematics::Transformation& T_WC,
                                 const std::vector<cv::KeyPoint>* keypoints) {
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIndex < numCameras_,
                        "Camera index exceeds number of cameras.")
  std::lock_guard<std::mutex> lock(*featureDetectorMutexes_[cameraIndex]);

  // check there are no keypoints here
  OKVIS_ASSERT_TRUE(Exception, keypoints == nullptr, "external keypoints currently not supported")

  if (usingXFeat()) {
    // ADR-0040: XFeat replaces BRISK detect+describe entirely. Branch out
    // before the BRISK-only camera-awareness / extraction-direction setup
    // below (T_WC gravity alignment is a BRISK descriptor concern).
    return detectAndDescribeXFeat(cameraIndex, frameOut);
  }

  // hack: also initialise those maps for camera-aware extraction
  for (size_t i = 0; i < numCameras_; ++i) {
    if(!std::static_pointer_cast<cv::BriskDescriptorExtractor>(
         descriptorExtractors_.at(i))->isCameraAware()) {
      cv::Mat rays;
      cv::Mat imageJacobians;
      bool success = frameOut->geometry(i)->getCameraAwarenessMaps(rays, imageJacobians);
      OKVIS_ASSERT_TRUE(Exception, success, "camera awareness maps not initialised");
      if (std::strcmp(frameOut->geometry(i)->type().c_str(), "EUCMCamera") == 0){
        std::static_pointer_cast<cv::BriskDescriptorExtractor>(descriptorExtractors_.at(i))->setCameraProperties(
                rays, imageJacobians, float(std::static_pointer_cast<const cameras::EucmCamera>(frameOut->geometry(i))->focalLengthU()));
      }
      else{
        std::static_pointer_cast<cv::BriskDescriptorExtractor>(descriptorExtractors_.at(i))->setCameraProperties(
                rays, imageJacobians, float(std::static_pointer_cast<const cameras::PinholeCameraBase>(frameOut->geometry(i))->focalLengthU()));
      }
    }
  }

  // ExtractionDirection == gravity direction in camera frame
  Eigen::Vector3d g_in_W(0, 0, -1);
  Eigen::Vector3d extractionDir = T_WC.inverse().C() * g_in_W;
  std::static_pointer_cast<cv::BriskDescriptorExtractor>(descriptorExtractors_[cameraIndex])
      ->setExtractionDirection(
        cv::Vec3f(float(extractionDir[0]),float(extractionDir[1]),float(extractionDir[2])));

  frameOut->setDetector(cameraIndex, featureDetectors_[cameraIndex]);
  frameOut->setExtractor(cameraIndex, descriptorExtractors_[cameraIndex]);
#ifdef OKVIS_USE_NN
  frameOut->setNetwork(cameraIndex, networks_[cameraIndex]);
#endif

  // detect
  const auto t0 = std::chrono::steady_clock::now();
  frameOut->detect(cameraIndex);

  // extract
  frameOut->describe(cameraIndex);

  // precompute backprojections
  frameOut->computeBackProjections(cameraIndex);

  {
    // T-0113 stats: BRISK is timed per camera frame (the cameras run in
    // parallel threads), XFeat per stereo pair — see writeStatsJson.
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();
    std::lock_guard<std::mutex> lock(statsMutex_);
    stats_.frontendMs.push_back(ms);
    stats_.keypoints += frameOut->numKeypoints(cameraIndex);
    stats_.frames += 1;
  }
  return true;
}

Frontend::Stats Frontend::stats() const {
  std::lock_guard<std::mutex> lock(statsMutex_);
  return stats_;
}

bool Frontend::writeStatsJson(const std::string& path) const {
  const Stats s = stats();
  std::vector<double> ms = s.frontendMs;
  std::sort(ms.begin(), ms.end());
  auto pct = [&ms](double p) {
    return ms.empty() ? 0.0 : ms[std::min(ms.size() - 1, size_t(p * double(ms.size())))];
  };
  const double mean = ms.empty() ? 0.0
      : std::accumulate(ms.begin(), ms.end(), 0.0) / double(ms.size());
  auto ratio = [](uint64_t a, uint64_t b) { return b ? double(a) / double(b) : 0.0; };
  std::ofstream f(path);
  if (!f.good()) {
    LOG(ERROR) << "cannot write frontend stats to " << path;
    return false;
  }
  f << std::setprecision(6) << std::fixed << "{\n"
    << "  \"engine_loaded\": " << (usingXFeat() ? 1 : 0) << ",\n"
    << "  \"float_descriptors\": " << (floatDescriptors_ ? 1 : 0) << ",\n"
    << "  \"frontend\": \"" << (usingXFeat() ? "xfeat" : "brisk") << "\",\n"
    << "  \"engine\": \"" << (usingXFeat() ? xfeatParams_.engine : "") << "\",\n"
    << "  \"frontend_calls\": " << ms.size() << ",\n"
    << "  \"frontend_call_unit\": \"" << (usingXFeat() && numCameras_ == 2 ? "stereo_pair" : "camera_frame") << "\",\n"
    << "  \"mean_frontend_ms\": " << mean << ",\n"
    << "  \"p99_frontend_ms\": " << pct(0.99) << ",\n"
    << "  \"max_frontend_ms\": " << (ms.empty() ? 0.0 : ms.back()) << ",\n"
    << "  \"frames\": " << s.frames << ",\n"
    << "  \"mean_keypoints_per_frame\": " << ratio(s.keypoints, s.frames) << ",\n"
    << "  \"stereo_calls\": " << s.stereoCalls << ",\n"
    << "  \"mean_stereo_matches\": " << ratio(s.stereoMatches, s.stereoCalls) << ",\n"
    << "  \"stereo_lighterglue_fallbacks\": " << s.stereoLgFallbacks << ",\n"
    << "  \"keyframe_match_calls\": " << s.keyframeCalls << ",\n"
    << "  \"mean_keyframe_matches\": " << ratio(s.keyframeMatches, s.keyframeCalls) << ",\n"
    << "  \"loop_closures\": " << s.loopClosures << "\n"
    << "}\n";
  LOG(INFO) << "frontend stats written to " << path;
  return true;
}

bool Frontend::verifyRecognisedPlace(const Estimator &estimator,
                                     const okvis::ViParameters &params,
                                     const std::shared_ptr<const MultiFrame>framesInOut,
                                     const std::shared_ptr<const MultiFrame> oldFrame,
                                     kinematics::Transformation &T_Sold_Snew,
                                     Eigen::Matrix<double, 6, 6>& H,
                                     int minInliers)
{
  // geometric verification:
  cameras::NCameraSystem::DistortionType distortionType = params.nCameraSystem.distortionType(0);
  std::map<LandmarkId, std::vector<const uchar *>> descriptors;
  AlignedMap<LandmarkId, Eigen::Vector4d> landmarks;
  int ctr = 0;
  opengv::absolute_pose::LoopclosureNoncentralAbsoluteAdapter::Points points;
  std::map<KeypointIdentifier, uint64_t> matches;

  // find matchable points
  TimerSwitchable loopClosureDescriptorMatchingTimer("2.04 loop closure descriptor matching");
  for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im) {
    for (size_t kOld = 0; kOld < oldFrame->numKeypoints(im); ++kOld) {
      uint64_t lmId = oldFrame->landmarkId(im, kOld);
      if (lmId == 0) {
        continue;
      }
      // get the 3D points / descriptors from old frame
      const uchar *oldDescripor = oldFrame->keypointDescriptor(im, kOld);
      Eigen::Vector4d landmark;
      bool isInitialised = false;
      oldFrame->getLandmark(im, kOld, landmark, isInitialised);
      if (!isInitialised)
        continue;
      if (landmark.norm() < 1.0e-12) {
        continue; // bit of a hack, signals there was no associated 3d point
      }
#ifdef OKVIS_USE_NN
      if (params.frontend.use_cnn && oldFrame->isClassified(im)) {
        // make sure not to use sky or person points here
        cv::Mat classification;
        oldFrame->getClassification(im, kOld, classification);
        if (classification.at<float>(10) > 3.5f) { // Sky
          continue;
        }
        if (classification.at<float>(11) > 53.5f) { // Person
          continue;
        }
      }
#endif
      auto iter = descriptors.find(LandmarkId(lmId));
      if (iter != descriptors.end()) {
        // just add descriptor
        iter->second.push_back(oldDescripor);
      } else {
        descriptors[LandmarkId(lmId)].push_back(oldDescripor);
        landmarks[LandmarkId(lmId)] = landmark;
      }
    }
  }

  // match. T-0120: on the XFeat path with a LighterGlue engine, match old->new
  // per camera with LighterGlue (KB 02 decision table: loop verification is the
  // wide-baseline stage) and keep the proposals that land on an initialised
  // landmark; otherwise the brute-force descriptor search below.
  bool lighterGlueMatched = false;
  if (floatDescriptors() && usingXFeat()) {
    std::vector<int> newForOld;
    for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im) {
      if (!lighterGluePairProposals(*oldFrame, im, *framesInOut, im, newForOld)) {
        break;
      }
      lighterGlueMatched = true;
      for (size_t kOld = 0; kOld < oldFrame->numKeypoints(im); ++kOld) {
        const int kNew = newForOld[kOld];
        const uint64_t lmId = oldFrame->landmarkId(im, kOld);
        if (kNew < 0 || lmId == 0 || !landmarks.count(LandmarkId(lmId))) {
          continue;
        }
        const KeypointIdentifier kid(framesInOut->id(), im, size_t(kNew));
        if (matches.count(kid)) {
          continue;
        }
        ctr++;
        points[lmId] = landmarks.at(LandmarkId(lmId));
        matches[kid] = lmId;
      }
    }
  }
  for (auto iter = landmarks.begin(); !lighterGlueMatched && iter != landmarks.end(); ++iter) {
    for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im) {

      if(framesInOut->numKeypoints(im) == 0) {
        continue;
      }
      
      const uchar *ddata = framesInOut->keypointDescriptor(im, 0);
      const size_t K = framesInOut->numKeypoints(im);
      const size_t descBytes = descriptorBytes();
      double distMin = briskMatchingThreshold_;
      size_t kMin = 0;
      for (const unsigned char* oldDescripor : descriptors.at(iter->first)) {
        for (size_t k = 0; k < K; ++k) {
          const double dist = descriptorDist(ddata + descBytes * k, oldDescripor);
          if (dist < distMin) {
            distMin = dist;
            kMin = k;
          }
        }
      }
      // now get best match
      if (distMin < briskMatchingThreshold_) {
        ctr++;
        const KeypointIdentifier kid(framesInOut->id(), im, kMin);
        points[iter->first.value()] = iter->second;
        matches[kid] = iter->first.value();
      }
    }
  }

  loopClosureDescriptorMatchingTimer.stop();

  if (ctr < minInliers || points.size() < 8) {
    return false;
  }

  // run 3d2d RANSAC (okvis/LoopClosureGates.hpp, T-0120: shared with the gtest).
  // Threshold: upstream's 16 (~5 px) for BRISK; vpr.reproj_px on the VPR path.
  const double ransacThreshold = (floatDescriptors() && vprLoop_)
      ? loopclosure::ransacThresholdFromPixels(vprParams_.reproj_px, xfeatParams_.keypoint_size)
      : 16.0;
  TimerSwitchable ransacLoopClosureTimer("2.05 loop closure ransacking");
  std::vector<bool> inliers;
  std::vector<size_t> ransacCamIndices, ransacKeypointIndices;
  const int numInliers = loopclosure::ransacAbsolutePose(
      points, matches, framesInOut, ransacThreshold, 50, T_Sold_Snew, inliers,
      ransacCamIndices, ransacKeypointIndices);
  ransacLoopClosureTimer.stop();
  const size_t numCorrespondences = inliers.size();
  if (numCorrespondences < 7) {
    return false;
  }
  // Inlier-ratio floor: upstream's 0.7 for BRISK brute-force matches; on the VPR
  // path the correspondences are LighterGlue proposals on ~6 px-noise fisheye
  // XFeat keypoints (T-0113 open item), so at reproj_px 2 the ratio is ~0.47
  // for TRUE revisits (loop_verify_offline sweep, T-0120) — min_inlier_ratio.
  const double minInlierRatio = (floatDescriptors() && vprLoop_) ? vprParams_.min_inlier_ratio : 0.7;
  const double inlierRatio = double(numInliers) / double(numCorrespondences);
  if (numInliers < minInliers || inlierRatio < minInlierRatio) {
    return false;
  }

  // check distinciveness of survived matches (BRISK bit statistics — skipped
  // with float descriptors; unreachable there anyway, the DBoW paths are off)
  if (!floatDescriptors()) {
  float sum = 0.0;
  for (size_t im = 0; im < numCameras_; ++im) {
    Eigen::Matrix<float, Eigen::Dynamic, 48 * 8> descriptorMatrix(numInliers, 48 * 8);
    int inlierCtr = 0;
    int ctr2 = 0;
    for (const auto &match : matches) {
      if (match.first.cameraIndex != im) {
        ctr2++;
        continue;
      }
      const uchar *desc = framesInOut->keypointDescriptor(match.first.cameraIndex,
                                                          match.first.keypointIndex);
      if (inliers[ctr2]) {
        for (size_t b = 0; b < 48; b++) {
          for (size_t c = 0; c < 8; c++) {
            if (desc[b] & (1 << c)) {
              descriptorMatrix(inlierCtr, b * 8 + c) = 1.0;
            } else {
              descriptorMatrix(inlierCtr, b * 8 + c) = 0.0;
            }
          }
        }
        inlierCtr++;
      }
      ctr2++;
    }
    if (inlierCtr > 0) {
      descriptorMatrix.conservativeResize(inlierCtr, 48 * 8);
      Eigen::Matrix<float, 1, 48 * 8> stdev
        = ((descriptorMatrix.rowwise() - descriptorMatrix.colwise().mean()).colwise().squaredNorm()
           / (descriptorMatrix.rows() - 1))
            .cwiseSqrt();
      sum += float(inlierCtr) * stdev.sum();
    }
  }

  const float avg = sum / float(numInliers);
  if (avg < 182.0 && numInliers < 20) {
    LOG(INFO) << framesInOut->id() << "->" << oldFrame->id() << " : "
              << "Rejecting loop closure due to indistincive descriptors (" << avg << ")";
    return false;
  }
  }

  // refine
  TimerSwitchable loopClosureRefinementTimer("2.06 loop closure pose refinement");
  const uint64_t frameId = framesInOut->id();

  // set up ceres problem
  ::ceres::Problem::Options problemOptions;
  problemOptions.manifold_ownership = ::ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
  problemOptions.loss_function_ownership = ::ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
  problemOptions.cost_function_ownership = ::ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
  ::ceres::Problem quickSolver(problemOptions);
  ::ceres::CauchyLoss cauchyLoss(3);
  ceres::PoseManifold pose6dParameterisation;
  ceres::HomogeneousPointManifold homogeneousPointParameterisation;

  // parameters: sensor pose and extrinsics
  std::shared_ptr<ceres::PoseParameterBlock> pose(new ceres::PoseParameterBlock(T_Sold_Snew, 1));
  quickSolver.AddParameterBlock(pose->parameters(), 7, &pose6dParameterisation);
  std::vector<std::shared_ptr<ceres::PoseParameterBlock>> extrinsics;
  for (size_t i = 0; i < framesInOut->numFrames(); ++i) {
    kinematics::Transformation T_SC = estimator.extrinsics(StateId(frameId), uchar(i));
    extrinsics.push_back(
      std::shared_ptr<ceres::PoseParameterBlock>(new ceres::PoseParameterBlock(T_SC, i + 2)));
    quickSolver.AddParameterBlock(extrinsics.back()->parameters(), 7, &pose6dParameterisation);
    quickSolver.SetParameterBlockConstant(extrinsics.back()->parameters());
  }
  const size_t landmarkIdOffset = framesInOut->numFrames() + 2;
  std::map<size_t, std::shared_ptr<ceres::ParameterBlock>> lms;
  std::vector<std::shared_ptr<ceres::ReprojectionError2dBase>> reprojectionErrors;
  std::vector<std::pair<const double *, const double *>> extrinsicsAndLandmarks;

  // add error terms and create landmarks if necessary
  for (size_t k = 0; k < numCorrespondences; ++k) {
    if (inliers[k]) {
      // get the landmark id:
      size_t camIdx = ransacCamIndices[k];
      size_t keypointIdx = ransacKeypointIndices[k];
      KeypointIdentifier kid(frameId, camIdx, keypointIdx);
      uint64_t lmId = matches.at(kid);
      std::shared_ptr<ceres::ParameterBlock> landmark;
      if (!lms.count(lmId + landmarkIdOffset)) {
        landmark.reset(
          new ceres::HomogeneousPointParameterBlock(points.at(lmId), lmId + landmarkIdOffset));
        quickSolver.AddParameterBlock(landmark->parameters(), 4, &homogeneousPointParameterisation);
        quickSolver.SetParameterBlockConstant(landmark->parameters());
        lms[lmId + landmarkIdOffset] = landmark;
      } else {
        landmark = lms.at(lmId + landmarkIdOffset);
      }
      Eigen::Vector2d kp;
      if (framesInOut->getKeypoint(camIdx, keypointIdx, kp)) {
        double size = 1.0;
        framesInOut->getKeypointSize(camIdx, keypointIdx, size);
        switch (distortionType) {
        case okvis::cameras::NCameraSystem::RadialTangential: {
          std::shared_ptr<
            ceres::ReprojectionError<cameras::PinholeCamera<cameras::RadialTangentialDistortion>>>
            reprojectionError(new ceres::ReprojectionError<
                              cameras::PinholeCamera<cameras::RadialTangentialDistortion>>(
              framesInOut->geometryAs<cameras::PinholeCamera<cameras::RadialTangentialDistortion>>(
                camIdx),
              camIdx,
              kp,
              64.0 / (size * size) * Eigen::Matrix2d::Identity()));
          reprojectionErrors.push_back(reprojectionError);
          quickSolver.AddResidualBlock(reprojectionError.get(),
                                       &cauchyLoss,
                                       pose->parameters(),
                                       landmark->parameters(),
                                       extrinsics[camIdx]->parameters());
          break;
        }
        case okvis::cameras::NCameraSystem::Equidistant: {
          std::shared_ptr<
            ceres::ReprojectionError<cameras::PinholeCamera<cameras::EquidistantDistortion>>>
            reprojectionError(
              new ceres::ReprojectionError<cameras::PinholeCamera<cameras::EquidistantDistortion>>(
                framesInOut->geometryAs<cameras::PinholeCamera<cameras::EquidistantDistortion>>(
                  camIdx),
                camIdx,
                kp,
                64.0 / (size * size) * Eigen::Matrix2d::Identity()));
          reprojectionErrors.push_back(reprojectionError);
          quickSolver.AddResidualBlock(reprojectionError.get(),
                                       &cauchyLoss,
                                       pose->parameters(),
                                       landmark->parameters(),
                                       extrinsics[camIdx]->parameters());
          break;
        }
        case okvis::cameras::NCameraSystem::RadialTangential8: {
          std::shared_ptr<
            ceres::ReprojectionError<cameras::PinholeCamera<cameras::RadialTangentialDistortion8>>>
            reprojectionError(new ceres::ReprojectionError<
                              cameras::PinholeCamera<cameras::RadialTangentialDistortion8>>(
              framesInOut->geometryAs<cameras::PinholeCamera<cameras::RadialTangentialDistortion8>>(
                camIdx),
              camIdx,
              kp,
              64.0 / (size * size) * Eigen::Matrix2d::Identity()));
          reprojectionErrors.push_back(reprojectionError);
          quickSolver.AddResidualBlock(reprojectionError.get(),
                                       &cauchyLoss,
                                       pose->parameters(),
                                       landmark->parameters(),
                                       extrinsics[camIdx]->parameters());
          break;
        }
        default:
          OKVIS_THROW(Exception, "Unsupported distortion type.")
          break;
        }
        extrinsicsAndLandmarks.push_back(
          std::pair<const double *, const double *>(landmark->parameters(),
                                                    extrinsics[camIdx]->parameters()));
      }
    }
  }

  // get solution
  ::ceres::Solver::Options solverOptions;
  ::ceres::Solver::Summary summary;
  solverOptions.num_threads = params.estimator.realtime_num_threads;
  solverOptions.max_num_iterations = params.estimator.realtime_max_iterations;
  solverOptions.minimizer_progress_to_stdout = false;
  ::ceres::Solve(solverOptions, &quickSolver, &summary);
  T_Sold_Snew = pose->estimate();

  // get uncertainty: lhs to be filled on the fly
  H = Eigen::Matrix<double, 6, 6>::Zero();
  int additionalOutliers = 0;
  for (size_t e = 0; e < reprojectionErrors.size(); ++e) {
    // fill lhs Hessian
    const double *pars[3];
    pars[0] = pose->parameters();
    pars[1] = extrinsicsAndLandmarks.at(e).first;
    pars[2] = extrinsicsAndLandmarks.at(e).second;
    const auto &reprojectionError = reprojectionErrors.at(e);
    double *jacobians[3];
    double *jacobiansMinimal[3];
    Eigen::Vector2d err;
    Eigen::Matrix<double, 2, 7> jacobian;
    Eigen::Matrix<double, 2, 6> jacobianMinimal;
    jacobians[0] = jacobian.data();
    jacobiansMinimal[0] = jacobianMinimal.data();
    jacobians[1] = nullptr;
    jacobiansMinimal[1] = nullptr;
    jacobians[2] = nullptr;
    jacobiansMinimal[2] = nullptr;
    reprojectionError->EvaluateWithMinimalJacobians(pars, err.data(), jacobians, jacobiansMinimal);  
    if (err.norm() > 3.0) {
      additionalOutliers++;
    } else {
      H += jacobianMinimal.transpose() * jacobianMinimal;
    }
  }

  const int numFinalInliers = numInliers-additionalOutliers;
  const double finalInlierRatio = double(numFinalInliers)
                             / double(numCorrespondences);
  if (numFinalInliers < minInliers || finalInlierRatio < minInlierRatio) {
    return false;
  }

  loopClosureRefinementTimer.stop();
  return true;
}

// filtered DBoW query result
int Frontend::getFilteredDBoWResult(const std::unique_ptr<DBoW> &dBow,
                                    const std::vector<std::vector<uchar>> &features,
                                    std::vector<std::pair<StateId, double>> &stateIds) const
{
  DBoW2::QueryResults dBoWResult;
  dBow->database.query(features, dBoWResult, -1); // get all matches
  DBoW2::QueryResults dBoWResultOrig = dBoWResult;
  // sort ascending -- we want to match oldest...
  std::sort(dBoWResult.begin(),
            dBoWResult.end(),
            [](const DBoW2::Result &lhs, const DBoW2::Result &rhs) { return lhs.Id < rhs.Id; });

  // nonmax suppression
  std::set<uint64_t> ids;
  std::set<uint64_t> suppressedIds;
  const size_t numKeyframes = dBoWResultOrig.size();
  int nonmaxRadius = 5;
  for (size_t f = 0; f < numKeyframes; ++f) {
    const double score = dBoWResultOrig[f].Score;
    const uint64_t id = dBoWResultOrig[f].Id;
    if (id >= dBoWResult.size())
      continue;
    if (score < 0.375) {
      break;
    }
    // check suppressed:
    if (suppressedIds.count(f)) {
      continue;
    }
    // check maximum
    bool isMax = true;
    for (int a = std::max(0, int(id) - nonmaxRadius);
         a <= (std::min(int(numKeyframes-1), int(id) + nonmaxRadius));
         ++a) {
      if (dBoWResult[a].Score > score) {
        isMax = false;
      }
    }
    if (!isMax) {
      continue;
    }

    // suppress
    for (int a = std::max(0, int(id) - nonmaxRadius);
         a <= (std::min(int(numKeyframes-1), int(id) + nonmaxRadius));
         ++a) {
      suppressedIds.insert(a);
    }

    // use
    ids.insert(id);
  }

  for (size_t id : ids) {

    // start with oldest keyframe match
    const double p = dBoWResult.at(id).Score;

    // get old multiframe
    uint64_t poseId = dBow->poseIds.at(id);

    // output
    stateIds.push_back(std::make_pair(StateId(poseId),p));
  }

  return stateIds.size();
}

// Matching as well as initialization of landmarks and state.
bool Frontend::dataAssociationAndInitialization(
    Estimator &estimator, const okvis::ViParameters& params,
    std::shared_ptr<okvis::MultiFrame> framesInOut, bool kfPrior, bool* asKeyframe) {

  // match new keypoints to existing landmarks/keypoints
  // initialise new landmarks (states)
  // outlier rejection by consistency check
  // RANSAC (2D2D / 3D2D)
  // decide keyframe
  // left-right stereo match & init

  // find distortion type
  cameras::NCameraSystem::DistortionType distortionType = params.nCameraSystem.distortionType(0);
  for (size_t i = 1; i < params.nCameraSystem.numCameras(); ++i) {
    OKVIS_ASSERT_TRUE(Exception, distortionType == params.nCameraSystem.distortionType(i),
                      "mixed frame types are not supported yet")
  }
  int num3dMatches = 0;

  // first frame? (did do addStates before, so 1 frame minimum in estimator)
  double trackingQuality = 1.0;
  if (estimator.numFrames() > 1) {

    // match to all landmarks
    TimerSwitchable matchMapTimer("2.01 match to map");
    switch (distortionType) {
      case okvis::cameras::NCameraSystem::RadialTangential: {
        num3dMatches = matchToMap<cameras::PinholeCamera<cameras::RadialTangentialDistortion>>(
            estimator, params, framesInOut->id());
        break;
      }
      case okvis::cameras::NCameraSystem::Equidistant: {
        num3dMatches = matchToMap<cameras::PinholeCamera<cameras::EquidistantDistortion>>(
            estimator, params, framesInOut->id());
        break;
      }
      case okvis::cameras::NCameraSystem::RadialTangential8: {
        num3dMatches = matchToMap<cameras::PinholeCamera<cameras::RadialTangentialDistortion8>>(
            estimator, params, framesInOut->id());
        break;
      }
      case okvis::cameras::NCameraSystem::NoDistortion: {
        num3dMatches = matchToMap<cameras::EucmCamera>(estimator, params, framesInOut->id());
        break;
      }
      default:
        OKVIS_THROW(Exception, "Unsupported distortion type.")
        break;
    }
    matchMapTimer.stop();
    {
      std::lock_guard<std::mutex> lock(statsMutex_);
      stats_.keyframeMatches += uint64_t(std::max(0, num3dMatches));
      ++stats_.keyframeCalls;
    }

    // check tracking quality
    trackingQuality = estimator.trackingQuality(StateId(framesInOut->id()));
    if (trackingQuality < 0.01) {
      if(estimator.numFrames() == 2 && params.nCameraSystem.numCameras()==1) {
        // mono. we can't have matches at this point, so don't warn
      } else {
        if (num3dMatches >= 3) {
          LOG(WARNING) << "3d2d tracking weak: quality=" << trackingQuality
                       << ". Number of 3d2d-matches: " << num3dMatches;
        } else {
          LOG(WARNING) << "3d2d tracking lost. Number of 3d2d-matches: " << num3dMatches;
        }
      }
    }

    // do motion stereo
    if(!trackingLost_ || !isInitialized_) {
      bool rotationOnly = false;
      TimerSwitchable matchMotionStereoTimer("2.02 match motion stereo");
      switch (distortionType) {
        case okvis::cameras::NCameraSystem::RadialTangential: {
          matchMotionStereo<cameras::PinholeCamera<cameras::RadialTangentialDistortion>>(
              estimator, params, framesInOut->id(), rotationOnly);
          break;
        }
        case okvis::cameras::NCameraSystem::Equidistant: {
          matchMotionStereo<cameras::PinholeCamera<cameras::EquidistantDistortion>>(
              estimator, params, framesInOut->id(), rotationOnly);
          break;
        }
        case okvis::cameras::NCameraSystem::RadialTangential8: {
          matchMotionStereo<cameras::PinholeCamera<cameras::RadialTangentialDistortion8>>(
              estimator, params, framesInOut->id(), rotationOnly);
          break;
        }
        case okvis::cameras::NCameraSystem::NoDistortion: {
          matchMotionStereo<cameras::EucmCamera>(estimator, params, framesInOut->id(), rotationOnly);
          break;
        }
        default:
          OKVIS_THROW(Exception, "Unsupported distortion type.")
          break;
      }
      if(!rotationOnly || num3dMatches>5) {
        if(!isInitialized_) {
          isInitialized_ = true;
          LOG(INFO) << "Initialized!";
        }
      }
      trackingQuality = estimator.trackingQuality(StateId(framesInOut->id()));
      matchMotionStereoTimer.stop();
    }
    //OKVIS_ASSERT_TRUE(Exception, estimator.areLandmarksInFrontOfCameras(), "after match motion stereo")

    // keyframe decision, at the moment only landmarks that match with keyframe are initialised
    if(kfPrior){
      *asKeyframe = true;
     }
    else{
      *asKeyframe = doWeNeedANewKeyframe(estimator, framesInOut);
    }
  } else {
    *asKeyframe = true;  // first frame needs to be keyframe
  }

  // prepare features for place recognition. BRISK only: the DBoW vocabulary is
  // BRISK-trained and unusable with XFeat float descriptors — with the XFeat
  // frontend all DBoW paths (multi-session + loop closure) stay off (ADR-0040;
  // DINOv2/FAISS place recognition is the planned replacement).
  std::vector<std::vector<uchar>> features;
  if (floatDescriptors()) {
    // T-0112: place recognition returns no candidates on the float path; say so once.
    static std::once_flag logged;
    std::call_once(logged, [] {
      LOG(WARNING) << "float descriptors: DBoW2 place recognition / loop closure "
                      "disabled (BRISK vocabulary not applicable, ADR-0040)";
    });
  } else {
    features.resize(framesInOut->numKeypoints());
    // first, we are trying to match the database for loop closures
    int offset = 0;
    for (size_t im = 0; im < numCameras_; ++im) {
      for (size_t k = 0; k < framesInOut->numKeypoints(im); ++k) {
        features.at(k + offset).resize(48); // TODO: get 48 from feature
        memcpy(features.at(k + offset).data(),
               framesInOut->keypointDescriptor(im, k),
               48 * sizeof(uchar));
      }
      offset += framesInOut->numKeypoints(im);
    }
  }

  /*MULTI-SESSION AND MULTI-AGENT*/
  if (!floatDescriptors() && !estimator.isLoopClosing() && !estimator.isLoopClosureAvailable()
      && !estimator.needsFullGraphOptimisation() && isInitialized_) {
    for (uint64_t c = 0; c < componentDBows_.size(); ++c) {
      TimerSwitchable matchDBoWTimer0("2.3.0 multi-session and multi-agent place recognition");
      std::vector<std::pair<StateId, double>> stateIds;
      getFilteredDBoWResult(componentDBows_.at(c), features, stateIds);
      matchDBoWTimer0.stop();

      int attempts = 0;

      for (const auto & id : stateIds) {

        double p = id.second;

        // get old multiframe
        MultiFramePtr oldMultiFrame = components_.at(c).multiFrames_.at(id.first);

        // stop after some amount of attempts
        if (attempts > std::max(10, int(componentDBows_.at(c)->poseIds.size() / 20)))
          break;
        if (p > 0.4) {
          kinematics::Transformation T_Sold_Snew;
          Eigen::Matrix<double, 6, 6> H;
          if (!verifyRecognisedPlace(estimator,
                                     params,
                                     framesInOut,
                                     oldMultiFrame,
                                     T_Sold_Snew,
                                     H,
                                     40)) {
            attempts++;
            continue;
          }
          attempts++;

          estimator.T_AiS_[StateId(framesInOut->id())][c] =
            components_.at(c).fullGraph_->pose(id.first) * T_Sold_Snew;

          break;
        }
      }
    }
  }

  /*LOOP CLOSURES*/
  // T-0120 (mowe-nav-kb 06 §adapter, ADR-0042): on the float-descriptor path the
  // DBoW2 query is replaced by VPR retrieval (DINOv2 + VLAD, mowe_vpr) at keyframe
  // rate; candidates pass the Mahalanobis prior gate, LighterGlue + GP3P RANSAC
  // (verifyRecognisedPlace) and temporal consistency before the UNCHANGED
  // attemptLoopClosure / addLoopClosureFrame / landmark-revival path below.
  const bool vprMode = floatDescriptors();
  bool vprQueried = false;
  if (vprMode && vprLoop_ && *asKeyframe && !kfPrior) {
    vprEmbedCurrent(*framesInOut);
  }
  if(params.estimator.do_loop_closures && (!vprMode || vprLoop_) && !estimator.isLoopClosing()
      && !estimator.isLoopClosureAvailable()
      && !estimator.needsFullGraphOptimisation() && isInitialized_) {
    TimerSwitchable matchDBoWTimer("2.03 loop closure query");
    std::vector<std::pair<StateId, double>> stateIds;
    double pMin = params.estimator.p_dbow;
    int minInliers = 10;
    size_t maxAttempts = 10;
    if (vprMode) {
      pMin = vprParams_.score_min;
      minInliers = vprParams_.min_inliers;
#ifdef OKVIS_USE_MOWE_XFEAT
      if (vprLoop_->haveCurrentDesc && vprLoop_->currentDescFrameId == framesInOut->id()) {
        vprQueried = true;
        vprQuery(estimator, params, framesInOut, stateIds);
        maxAttempts = std::max(size_t(10), vprLoop_->entries.size() / 20);
      }
#endif
    } else {
      dBow();  // ensure the BRISK vocabulary is loaded (lazy, T-0112)
      getFilteredDBoWResult(dBow_, features, stateIds);
      maxAttempts = std::max(size_t(10), dBow().poseIds.size() / 20);
    }
    matchDBoWTimer.stop();
    TimerSwitchable attemptLoopClosureTimer("2.07 attempt loop closure", true);
    // nonmax suppression
    size_t attempts = 0;
    bool vprVerified = false;
    for(const auto & id : stateIds) {
      // start with oldest keyframe match
      const double p = id.second;
      // get old multiframe
      if(attempts > maxAttempts) break;
      if(p > pMin) {

        const std::shared_ptr<const MultiFrame> oldFrame = estimator.multiFrame(id.first);
        /// \todo move to separate thread
        // check if already existing loop closure or matching against current frame
        if(!oldFrame || !estimator.isPoseGraphFrame(id.first)
           || estimator.isLoopClosureFrame(id.first)
           || estimator.isRecentLoopClosureFrame(id.first)
           || !estimator.isPlaceRecognitionFrame(id.first)) {
          if (vprMode) ++loopStats_.candidatesSkippedState;
          continue;
        }
        // T-0120 prior gate (KB 06 defence #1): candidate pose vs the current VIO
        // estimate under a drift-model covariance. Without RTK this is the VIO's
        // own consistency; with ADR-0042 GNSS factors the same estimate carries it.
        if (vprMode) {
          loopclosure::PriorGateParams gate;
          gate.sigma_pos_floor_m = vprParams_.prior_sigma_pos_m;
          gate.drift_frac = vprParams_.prior_drift_frac;
          gate.sigma_rot_rad = vprParams_.prior_sigma_rot_deg * M_PI / 180.0;
          double pathLength = 0.0;
#ifdef OKVIS_USE_MOWE_XFEAT
          {
            const VprLoop& L = *vprLoop_;
            const kinematics::Transformation T_WS_now = estimator.pose(StateId(framesInOut->id()));
            const double toNow = L.haveLast ? (T_WS_now.r() - L.r_last).norm() : 0.0;
            pathLength = L.cumDist + toNow - L.entries.at(size_t(L.indexOfState.at(id.first.value()))).cumDist;
          }
#endif
          const double m = loopclosure::priorMahalanobis(
              estimator.pose(StateId(framesInOut->id())), estimator.pose(id.first), pathLength, gate);
          loopStats_.priorMahalanobis.push_back(m);
          if (m > vprParams_.prior_gate_sigma) {
            ++loopStats_.rejectedByPriorGate;
            LOG(INFO) << "VPR candidate " << id.first.value() << " for frame " << framesInOut->id()
                      << " rejected by prior gate: " << m << " sigma (score " << p
                      << ", path " << pathLength << " m)";
            continue;
          }
        }
        // verify with RANSAC and refine
        kinematics::Transformation T_Sold_Snew;
        Eigen::Matrix<double, 6, 6> H;
        const auto tVerify0 = std::chrono::steady_clock::now();
        const bool verified = verifyRecognisedPlace(estimator, params, framesInOut, oldFrame,
                                                    T_Sold_Snew, H, minInliers);
        if (vprMode) {
          loopStats_.verificationMs.push_back(std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - tVerify0).count());
        }
        if (!verified) {
          if (vprMode) ++loopStats_.rejectedByGeometry;
          attempts++;
          continue;
        }
        attempts++;
        // T-0120 temporal consistency (KB 06 defence #2): one vote per keyframe
        // query — the strongest verified candidate — must agree with the previous
        // consecutive_required-1 keyframes' verified candidates.
        if (vprMode) {
          vprVerified = true;
#ifdef OKVIS_USE_MOWE_XFEAT
          if (!vprLoop_->temporal.push(vprLoop_->indexOfState.at(id.first.value()))) {
            ++loopStats_.rejectedByTemporal;
            LOG(INFO) << "VPR candidate " << id.first.value() << " for frame " << framesInOut->id()
                      << " verified (score " << p << ") but waiting for temporal consistency";
            break;
          }
#endif
        }
        // enforce relative transformation
        attemptLoopClosureTimer.start();
        bool skipFullGraphOptimisation = false;
        const uint64_t frameId = framesInOut->id();
        bool loopClosureAttemptSuccessful =
            estimator.attemptLoopClosure(
              StateId(oldFrame->id()), StateId(frameId), T_Sold_Snew, H,
              skipFullGraphOptimisation,
              params.estimator.drift_percentage_heuristic);
        if(!loopClosureAttemptSuccessful) {
          LOG(INFO) << "unsuccessful loop closure frame "<< id.first.value();
          attemptLoopClosureTimer.stop();
          if (vprMode) {
            ++loopStats_.rejectedByEstimator;
            break;
          }
          continue;
        }
        attemptLoopClosureTimer.stop();
        // bring back old landmarks
        TimerSwitchable addLoopClosureTimer("2.08 add loop closure");
        std::set<LandmarkId> loopClosureLandmarks;
        estimator.addLoopClosureFrame(StateId(oldFrame->id()), loopClosureLandmarks,
                                      skipFullGraphOptimisation);
        addLoopClosureTimer.stop();
        // properly match against all loopClosure frame points
        TimerSwitchable matchLoopClosureTimer("2.09 match loop closure");
        int loopClosureMatches = 0;
        switch (distortionType) {
          case okvis::cameras::NCameraSystem::RadialTangential: {
            loopClosureMatches =
                matchToMap<cameras::PinholeCamera<cameras::RadialTangentialDistortion>>(
                    estimator, params, framesInOut->id(), &loopClosureLandmarks);
            break;
          }
          case okvis::cameras::NCameraSystem::Equidistant: {
            loopClosureMatches = matchToMap<cameras::PinholeCamera<cameras::EquidistantDistortion>>(
                estimator, params, framesInOut->id(), &loopClosureLandmarks);
            break;
          }
          case okvis::cameras::NCameraSystem::RadialTangential8: {
            loopClosureMatches =
                matchToMap<cameras::PinholeCamera<cameras::RadialTangentialDistortion8>>(
                    estimator, params, framesInOut->id(), &loopClosureLandmarks);
            break;
          }
          case okvis::cameras::NCameraSystem::NoDistortion: {
            loopClosureMatches = matchToMap<cameras::EucmCamera>(
                    estimator, params, framesInOut->id(), &loopClosureLandmarks);
            break;
          }
          default:
            OKVIS_THROW(Exception, "Unsupported distortion type.")
            break;
        }
        LOG(INFO) << "LOOP CLOSURE: current frame " << framesInOut->id()
                  << ", matching to keyframe " << oldFrame->id() << ", "
                  << loopClosureMatches << " matches, p=" << p << ".";
        {
          std::lock_guard<std::mutex> lock(statsMutex_);
          ++stats_.loopClosures;
          if (vprMode) ++loopStats_.loopsAccepted;
        }

        matchLoopClosureTimer.stop();
        // re-decide keyframe:
        if(kfPrior){
          *asKeyframe = true;
        }
        else{
          *asKeyframe = doWeNeedANewKeyframe(estimator, framesInOut);
        }

        break; // only consider oldest keyframe match.
      }
    }
#ifdef OKVIS_USE_MOWE_XFEAT
    if (vprMode && vprQueried && !vprVerified) {
      vprLoop_->temporal.miss();
    }
#endif
    // if keyframe, we add to relocalisation database (VPR: below, also while the
    // estimator is busy loop-closing — the database must not skip keyframes)
    if(!vprMode && *asKeyframe && !kfPrior) {
      dBow().database.add(features);
      dBow().poseIds.push_back(framesInOut->id());
    }
  }
  if (vprMode && vprLoop_ && *asKeyframe && !kfPrior) {
    vprAddCurrent(estimator, *framesInOut);
  }


#ifdef OKVIS_USE_NN
  // This needs to be after keyframe re-decision, otherwise we might delete a frame before the CNN
  // finishes. This could obviously be done in a smarter way though.
  if(params.frontend.use_cnn && isInitialized_) {
    if(*asKeyframe) {
      // clean up threads if still running
      std::set<StateId> toDelete;
      for(auto threads = cnnThreads_.begin(); threads != cnnThreads_.end(); ++threads) {
        bool deleting = true;
        for(size_t i=0; i<threads->second.size(); ++i) {
          //if(threads->second[i] && estimator.multiFrame(StateId(threads->first))->isClassified(i)) {
            threads->second[i]->join();
            delete threads->second[i];
            threads->second[i] = nullptr;
          //} else {
          //  deleting = false;
          //}
        }
        if(deleting) {
          toDelete.insert(StateId(threads->first));
        }
      }
      for(const auto & deleting : toDelete) {
        cnnThreads_.erase(deleting);
      }

      // launch classification in background
      cnnThreads_[StateId(framesInOut->id())] = std::vector<std::thread*>(
            params.nCameraSystem.numCameras(), nullptr);
      for (size_t i = 0; i < params.nCameraSystem.numCameras(); ++i) {
        cnnThreads_[StateId(framesInOut->id())].at(i) =
            new std::thread(&MultiFrame::computeClassifications,framesInOut.get(), i,
                      64*(framesInOut->image(i).cols/64), 64*(framesInOut->image(i).rows/64));
      }
    }
  }
#else
  OKVIS_ASSERT_TRUE(Exception, !params.frontend.use_cnn,
                    "Requested CNN classification, but not compiled with USE_NN option.")
#endif

  // do stereo match -- get new landmarks only when this is a keyframe
  if(*asKeyframe) {
    TimerSwitchable matchStereoTimer("2.10 match stereo");
    switch (distortionType) {
      case okvis::cameras::NCameraSystem::RadialTangential: {
        matchStereo<cameras::PinholeCamera<cameras::RadialTangentialDistortion>>(
              estimator, framesInOut, params, *asKeyframe);
        break;
      }
      case okvis::cameras::NCameraSystem::Equidistant: {
        matchStereo<cameras::PinholeCamera<cameras::EquidistantDistortion>>(
              estimator, framesInOut, params, *asKeyframe);
        break;
      }
      case okvis::cameras::NCameraSystem::RadialTangential8: {
        matchStereo<okvis::cameras::PinholeCamera<cameras::RadialTangentialDistortion8>>(
              estimator, framesInOut, params, *asKeyframe);
        break;
      }
      case okvis::cameras::NCameraSystem::NoDistortion: {
        matchStereo<okvis::cameras::EucmCamera>(
                estimator, framesInOut, params, *asKeyframe);
        break;
        }
      default:
        OKVIS_THROW(Exception, "Unsupported distortion type.")
        break;
    }
    matchStereoTimer.stop();
    //OKVIS_ASSERT_TRUE(Exception, estimator.areLandmarksInFrontOfCameras(), "after stereo")

  }

  // remove outliers, as the last matching step may have introduced some:
  switch (distortionType) {
  case okvis::cameras::NCameraSystem::RadialTangential: {
    removeOutliers<cameras::PinholeCamera<cameras::RadialTangentialDistortion>>(
      estimator, params.nCameraSystem, framesInOut);
    break;
  }
  case okvis::cameras::NCameraSystem::Equidistant: {
    removeOutliers<cameras::PinholeCamera<cameras::EquidistantDistortion>>(
      estimator, params.nCameraSystem, framesInOut);
    break;
  }
  case okvis::cameras::NCameraSystem::RadialTangential8: {
    removeOutliers<okvis::cameras::PinholeCamera<cameras::RadialTangentialDistortion8>>(
      estimator, params.nCameraSystem, framesInOut);
    break;
  }
  default:
    OKVIS_THROW(Exception, "Unsupported distortion type.")
    break;
  }


  // remove outliers, as the last matching step may have introduced some:
  switch (distortionType) {
  case okvis::cameras::NCameraSystem::RadialTangential: {
    removeOutliers<cameras::PinholeCamera<cameras::RadialTangentialDistortion>>(
      estimator, params.nCameraSystem, framesInOut);
    break;
  }
  case okvis::cameras::NCameraSystem::Equidistant: {
    removeOutliers<cameras::PinholeCamera<cameras::EquidistantDistortion>>(
      estimator, params.nCameraSystem, framesInOut);
    break;
  }
  case okvis::cameras::NCameraSystem::RadialTangential8: {
    removeOutliers<okvis::cameras::PinholeCamera<cameras::RadialTangentialDistortion8>>(
      estimator, params.nCameraSystem, framesInOut);
    break;
  }
  default:
    OKVIS_THROW(Exception, "Unsupported distortion type.")
    break;
  } //ToDo:EUCM

#ifdef OKVIS_USE_NN
  if(params.frontend.use_cnn) {
    // remove matches into dynamic areas
    // get all landmarks
    MapPoints pointMap;
    estimator.getLandmarks(pointMap);
    for(MapPoints::iterator it = pointMap.begin(); it != pointMap.end(); ++it) {
      bool remove = false;
      if(it->second.classification == 10 || it->second.classification == 11) {
        remove = true;
      } else {
      for(auto& obs : it->second.observations) {
        if(!estimator.isKeyframe(StateId(obs.frameId))) continue;
        auto frame = estimator.multiFrame(StateId(obs.frameId));
        if(!frame->isClassified(obs.cameraIndex)) continue;
        cv::Mat classification;
        if(frame->getClassification(obs.cameraIndex, obs.keypointIndex, classification)) {
          if(classification.at<float>(10) > 3.5f) { // Sky
            remove = true;
            Eigen::Vector2d kpt;
            frame->getKeypoint(obs.cameraIndex, obs.keypointIndex, kpt);
            estimator.setLandmarkClassification(it->first, 10);
            break;
          }
          if(classification.at<float>(11) > 53.5f) { // Person
            remove = true;
            estimator.setLandmarkClassification(it->first, 11);
            break;
          }
        }
      }
      }
      if(remove) {
        std::set<KeypointIdentifier> observations = it->second.observations;
        for(auto& obs : observations) {
          estimator.setObservationInformation(
                StateId(obs.frameId), obs.cameraIndex, obs.keypointIndex,
                Eigen::Matrix2d::Identity()*0.0001);
        }
      }
    }
  }
#endif
  estimator.cleanUnobservedLandmarks();

  return trackingQuality >= 0.01;
}

// Propagates pose, speeds and biases with given IMU measurements.
bool Frontend::propagation(
    const okvis::ImuMeasurementDeque& imuMeasurements, const okvis::ImuParameters& imuParams,
    okvis::kinematics::Transformation& T_WS_propagated, okvis::SpeedAndBias& speedAndBiases,
    const okvis::Time& t_start, const okvis::Time& t_end, Eigen::Matrix<double, 15, 15>* covariance,
    Eigen::Matrix<double, 15, 15>* jacobian) const {

  if (imuMeasurements.size() < 2) {
    LOG(WARNING) << "- Skipping propagation as only one IMU measurement has been given to frontend."
                 << " Normal when starting up.";
    return 0;
  }
  int measurements_propagated =
      okvis::ceres::ImuError::propagation(imuMeasurements, imuParams, T_WS_propagated,
                                          speedAndBiases, t_start, t_end, covariance, jacobian);

  return measurements_propagated > 0;
}

void Frontend::endCnnThreads() {
  for(auto & threads : cnnThreads_) {
    for(auto & thread : threads.second) {
      if(thread) {
        thread->join();
        delete thread;
        thread = nullptr;
      }
    }
  }
}

void Frontend::clear()
{
  endCnnThreads();
  isInitialized_ = false;        // Is the pose initialised?
  if (dBow_) {
    dBow_->database.clear();
    dBow_->poseIds.clear(); // Store the multiframe IDs corresponsind to the dBow ones
  }
  trackingLost_ = false; // Is the tracking currently lost?
}

// Decision whether a new frame should be keyframe or not.
bool Frontend::doWeNeedANewKeyframe(const Estimator &estimator,
                                    std::shared_ptr<okvis::MultiFrame> currentFrame) {
  if (estimator.numFrames() < 4) {
    // just starting, so yes, we need this as a new keyframe
    return true;
  }

  if (!isInitialized_) return false;

  int intersectionCount = 0;
  int unionCount = 0;

  size_t numKeypoints = 0;

  // go through all the frames and try to match the initialized keypoints
  std::set<uint64_t> lmIds;
  for (size_t im = 0; im < currentFrame->numFrames(); ++im) {
    const int rows = currentFrame->image(im).rows/10;
    const int cols = currentFrame->image(im).cols/10;

    cv::Mat matches = cv::Mat::zeros(rows, cols, CV_8UC1);
    cv::Mat detections = cv::Mat::zeros(rows, cols, CV_8UC1);

    const size_t numB = currentFrame->numKeypoints(im);
    numKeypoints += numB;
    const double radius = double(std::min(rows,cols))*kptrad;
    cv::KeyPoint keypoint;
    for (size_t k = 0; k < numB; ++k) {
      currentFrame->getCvKeypoint(im, k, keypoint);
      cv::circle(detections, keypoint.pt*0.1, int(radius), cv::Scalar(255), cv::FILLED);
      uint64_t lmId = currentFrame->landmarkId(im, k);
      if (lmId != 0) {
        cv::circle(matches, keypoint.pt*0.1, int(radius), cv::Scalar(255), cv::FILLED);
        lmIds.insert(lmId);
      }
    }

    // IoU
    cv::Mat intersectionMask, unionMask;
    cv::bitwise_and(matches, detections, intersectionMask);
    cv::bitwise_or(matches, detections, unionMask);
    intersectionCount += cv::countNonZero(intersectionMask);
    unionCount += cv::countNonZero(unionMask);
  }

  double overlap = double(intersectionCount)/double(unionCount);

  std::set<StateId> allFrames = estimator.keyFrames();
  allFrames.insert(estimator.loopClosureFrames().begin(), estimator.loopClosureFrames().end());
  for(size_t age = 0; age < estimator.numFrames(); ++age) {
    auto id = estimator.stateIdByAge(age);
    if(!estimator.isInImuWindow(id)) {
      break;
    }
    if(estimator.isKeyframe(id)) {
      allFrames.insert(id);
    }
  }
  double overlapOthers = 0.0;
  for(auto frame : allFrames) {
    int intersectionCount = 0;
    int unionCount = 0;

    // go through all the frames and try to match the initialized keypoints
    auto otherFrame = estimator.multiFrame(frame);
    for (size_t im = 0; im < otherFrame->numFrames(); ++im) {
      const int rows = otherFrame->image(im).rows/10;
      const int cols = otherFrame->image(im).cols/10;

      cv::Mat matches = cv::Mat::zeros(rows, cols, CV_8UC1);
      cv::Mat detections = cv::Mat::zeros(rows, cols, CV_8UC1);

      const size_t numB = otherFrame->numKeypoints(im);

      const double radius = double(std::min(rows,cols))*kptrad;
      cv::KeyPoint keypoint;
      for (size_t k = 0; k < numB; ++k) {
        otherFrame->getCvKeypoint(im, k, keypoint);
        cv::circle(detections, keypoint.pt*0.1, int(radius), cv::Scalar(255), cv::FILLED);
        uint64_t lmId = otherFrame->landmarkId(im, k);
        if (lmId != 0 && lmIds.count(lmId)) {
          cv::circle(matches, keypoint.pt*0.1, int(radius), cv::Scalar(255), cv::FILLED);
        }
      }

      // IoU
      cv::Mat intersectionMask, unionMask;
      cv::bitwise_and(matches, detections, intersectionMask);
      cv::bitwise_or(matches, detections, unionMask);
      intersectionCount += cv::countNonZero(intersectionMask);
      unionCount += cv::countNonZero(unionMask);
    }

    overlapOthers = std::max(overlapOthers, double(intersectionCount)/double(unionCount));
  }

  overlap = std::min(overlapOthers, overlap);

  // take a decision
  if(numKeypoints < 7 * currentFrame->numFrames()) {
    // a respectable keyframe needs some detections...
    return false;
  }
  if (float(overlap) > keyframeInsertionOverlapThreshold_
      /*&& double(numMatches)/double(numKeypoints) > 0.35*/) {
    return false;
  } else {
    return true;
  }
}

// Match a new multiframe to existing keyframes
template <class CAMERA_GEOMETRY>
int Frontend::matchToMap(Estimator &estimator, const okvis::ViParameters& params,
                         const uint64_t currentFrameId,
                         const std::set<LandmarkId>* loopClosureLandmarksToUseExclusively) {

  if (estimator.numFrames() < 2) {
    // just starting, so yes, we need this as a new keyframe
    return 0;
  }

  // get all landmarks
  MapPoints pointMap;
  estimator.getLandmarks(pointMap);

  // these may be needed for loop-closure map fusion
  std::vector<LandmarkId> oldIds, newIds;

  // store the map to be matched
  std::vector<AlignedMap<LandmarkId, LandmarkToMatch>>
      landmarksToMatchVec(params.nCameraSystem.numCameras());

  // match
  int ctr = 0;
  std::vector<cv::Mat> descriptorPool(params.nCameraSystem.numCameras());
  kinematics::Transformation T_WS1 = estimator.pose(StateId(currentFrameId));
  double reprErr = 0.0;
  for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im) {

    // the current frame to match
    const MultiFramePtr multiFrame = estimator.multiFrame(StateId(currentFrameId));

    const double f = 0.5*(multiFrame->geometryAs<CAMERA_GEOMETRY>(im)->focalLengthU()
                            + multiFrame->geometryAs<CAMERA_GEOMETRY>(im)->focalLengthV());
    const double reprThreshold = params.imu.use ? 3.0+f*0.06 : 3.0+f*0.34;

    const size_t numKeypoints = multiFrame->numKeypoints(im);
    if(numKeypoints == 0) {
      continue; // no points -- bad!
    }

    // for checks if in image
    const double maxU = multiFrame->geometryAs<CAMERA_GEOMETRY>(im)->imageWidth() + reprThreshold;
    const double maxV = multiFrame->geometryAs<CAMERA_GEOMETRY>(im)->imageHeight() + reprThreshold;

    // prepare landmarks as visible in this frame
    AlignedMap<LandmarkId, LandmarkToMatch> landmarksToMatch;
    const kinematics::Transformation T_SC = *multiFrame->T_SC(im);
    const kinematics::Transformation T_WC1 = T_WS1 * T_SC;
    const kinematics::Transformation T_CW1 = T_WC1.inverse();

    const double focalLength =
        multiFrame->geometryAs<CAMERA_GEOMETRY>(im)->focalLengthU()
        + multiFrame->geometryAs<CAMERA_GEOMETRY>(im)->focalLengthV();

    // go through all landmarks
    const size_t numDescriptorsToKeep = 3; // use only best 3
    // Descriptor rows are raw bytes: 48 for BRISK, 256 for XFeat 64-D float
    // (the distance dispatch in descriptorDist() reinterprets accordingly).
    const int descBytes = int(descriptorBytes());
    descriptorPool[im] = cv::Mat(
        int(numDescriptorsToKeep)*pointMap.size(), descBytes, CV_8UC1);
    uchar* dataPtr = descriptorPool[im].data;
    for(MapPoints::const_iterator it = pointMap.begin(); it != pointMap.end(); ++it) {
      if(loopClosureLandmarksToUseExclusively) {
        if(!loopClosureLandmarksToUseExclusively->count(it->first)) {
          continue; // skip non-loop-closure points in this case
        }
      }
      // create landmark
      LandmarkToMatch landmarkToMatch;
      landmarkToMatch.is3d = false;

      // FoV check
      const Eigen::Vector4d hp_W = it->second.point;
      landmarkToMatch.p_W = hp_W.head<3>()/hp_W[3];
      const Eigen::Vector3d r_W = landmarkToMatch.p_W- T_WC1.r();
      const Eigen::Vector3d e_W = r_W.normalized();
      const double r = std::max(0.01, r_W.norm());

      const Eigen::Vector4d hp_C = T_CW1*hp_W;
      Eigen::Vector2d kp;
      const cameras::ProjectionStatus status = multiFrame->geometryAs<CAMERA_GEOMETRY>(im)
                                                 ->projectHomogeneous(hp_C, &kp);
      if(status == cameras::ProjectionStatus::Invalid
          || status == cameras::ProjectionStatus::Behind) {
        continue;
      }

      if (kp[0] < -reprThreshold)
        continue;
      if (kp[1] < -reprThreshold)
        continue;
      if (kp[0] > maxU)
        continue;
      if (kp[1] > maxV)
        continue;

      landmarkToMatch.projection = kp;

      // distinguish whether to consider as 3D point or not.
      const double quality = it->second.quality;

      // obtain map point descriptor, and do some pruning.
      std::vector<double> bestScores(numDescriptorsToKeep, 1.0);
      landmarkToMatch.descriptors = cv::Mat(
          int(numDescriptorsToKeep), descBytes, CV_8UC1, dataPtr);
      landmarkToMatch.e_W.resize(3,numDescriptorsToKeep);
      landmarkToMatch.r_W.resize(3,numDescriptorsToKeep);
      landmarkToMatch.kids.reserve(numDescriptorsToKeep);
      const LandmarkId landmarkId = it->first;
      size_t o=0;
      for (auto obsiter = it->second.observations.rbegin();
           obsiter != it->second.observations.rend();
           ++obsiter) {

        const KeypointIdentifier kid = *obsiter;

        // remove some descriptors that are unlikely to match
        const kinematics::Transformation T_SC_old =
            *multiFrame->T_SC(kid.cameraIndex);
        const kinematics::Transformation T_WS_old =
            estimator.pose(StateId(kid.frameId));
        const kinematics::Transformation T_WC_old = T_WS_old * T_SC_old;
        const Eigen::Vector3d r_W_old = hp_W.head<3>()/hp_W[3] - T_WC_old.r();

        // check if 3D
        if(!landmarkToMatch.is3d) {
          const Eigen::Vector3d r_close_W = r_W-(0.2/focalLength/quality*r_W_old);
          const double cosA = r_W.normalized().dot(r_close_W.normalized());
          if(cosA > cos(10.0/focalLength)) {
            landmarkToMatch.is3d = true;
          }
        }

        // over 35 degree viewpoint change
        const double cosViewpointChnage = e_W.dot(r_W_old.normalized());
        if(cosViewpointChnage < cos(0.6)
            && !loopClosureLandmarksToUseExclusively) {
          continue;
        }

        // scale change over 50%
        const double scaleChange = fabs(r-r_W_old.norm())/r;
        if((scaleChange > 0.5)
            && !loopClosureLandmarksToUseExclusively) {
          continue;
        }

        const double score = 0.5*(acos(cosViewpointChnage)/0.6+scaleChange/0.5);
        double worstScore = 0.0;
        size_t worstIdx = 0;
        // find location in buffer to write to
        for(size_t n=0; n<numDescriptorsToKeep; ++n) {
          if(bestScores[n] > worstScore) {
            worstScore = bestScores[n];
            worstIdx = n;
          }
        }
        // store if better
        if(score<bestScores[worstIdx]) {
          // copy over descriptors
          const MultiFramePtr oldFrame =
              estimator.multiFrame(StateId(kid.frameId));
          std::memcpy(
              landmarkToMatch.descriptors.data+size_t(descBytes)*worstIdx,
              oldFrame->keypointDescriptor(
                  kid.cameraIndex, kid.keypointIndex), size_t(descBytes));

          // remember some other stuff for efficiency
          Eigen::Vector3d e_C;
          oldFrame->getBackProjection(kid.cameraIndex, kid.keypointIndex, e_C);
          landmarkToMatch.e_W.col(worstIdx) = T_WC_old.C()*e_C.normalized();
          landmarkToMatch.r_W.col(worstIdx) = T_WC_old.r();
          landmarkToMatch.kids.push_back(kid);

          // remember which were used
          o = std::max(o, worstIdx);
          bestScores[worstIdx] = score;
        }
      }

      // crop unused bottom rows / right cols
      landmarkToMatch.descriptors =
          landmarkToMatch.descriptors(cv::Rect(0, 0, descBytes, o + 1));
      landmarkToMatch.e_W.conservativeResize(3,o + 1);
      landmarkToMatch.r_W.conservativeResize(3,o + 1);
      dataPtr += (o + 1) * size_t(descBytes);

      if(landmarkToMatch.descriptors.rows==0) {
        // no observations -- weird.
        continue;
      }

      // check classification
      if (it->second.classification == 10 || it->second.classification == 11) {
        landmarkToMatch.ignore = true;
      }

      // insert
      landmarksToMatch[landmarkId] = landmarkToMatch;
    }
    landmarksToMatchVec[im] = landmarksToMatch;

    // multithreaded matching
    const size_t num_matching_threads = size_t(params.frontend.num_matching_threads);

    std::vector<double> distances(numKeypoints,briskMatchingThreshold_);
    std::vector<LandmarkId> lmIds(numKeypoints);
    AlignedVector<Eigen::Vector4d> hps_W(numKeypoints, Eigen::Vector4d::Zero());
    std::vector<size_t> ctrs(num_matching_threads);
    std::vector<double> reprErrors(num_matching_threads);

    std::vector<std::thread*> threads(num_matching_threads, nullptr);
    for(size_t t = 0; t<num_matching_threads; ++t) {
      threads[t] = new std::thread(
          &Frontend::matchToMapByThread<CAMERA_GEOMETRY>, this, t, num_matching_threads,
              std::cref(estimator), std::cref(params), currentFrameId,
              loopClosureLandmarksToUseExclusively, std::cref(T_WS1),
              std::cref(landmarksToMatch), numKeypoints,
              std::cref(pointMap), im, std::cref(multiFrame), std::ref(distances),
              std::ref(lmIds), std::ref(hps_W), std::ref(ctrs), std::ref(reprErrors));
    }

    for(size_t t = 0; t<num_matching_threads; ++t) {
      threads[t]->join();
      delete threads[t];
      reprErr += reprErrors[t];
    }

    // now insert observations
    for(size_t k = 0; k < numKeypoints; ++k) {
      uint64_t previousId = multiFrame->landmarkId(im,k);
      if(lmIds[k].isInitialised()) {

        if(previousId && loopClosureLandmarksToUseExclusively) {
          // remove
          estimator.removeObservation(StateId(currentFrameId), im, k);
          oldIds.push_back(LandmarkId(previousId));
          newIds.push_back(lmIds[k]);
        }

        multiFrame->setLandmarkId(im, k, lmIds[k].value());
        estimator.addObservation<CAMERA_GEOMETRY>(
              lmIds[k], StateId(currentFrameId), im, k);
        if (landmarksToMatch[lmIds[k]].ignore) {
          estimator.setObservationInformation(StateId(currentFrameId), im, k,
                                              Eigen::Matrix2d::Identity()*0.00001);
        }
        ctr++;
      }
    }
  }
  //OKVIS_ASSERT_TRUE(Exception, estimator.areLandmarksInFrontOfCameras(), "before ransac")
  reprErr /= double(params.frontend.num_matching_threads * params.nCameraSystem.numCameras());

  // remove outliers -- initialise pose only without IMU or when matching with large repr. err.
  MultiFramePtr multiFrame = estimator.multiFrame(StateId(currentFrameId));
  int numInitIter = 2;
  const bool ransacRemoveOutliers = true;
  bool runRansac = !params.imu.use;
  const double f = 0.5*(multiFrame->geometryAs<CAMERA_GEOMETRY>(0)->focalLengthU()
                        + multiFrame->geometryAs<CAMERA_GEOMETRY>(0)->focalLengthV());
  const double strictReprThreshold = 3.0 + f*0.006;
  if (reprErr > strictReprThreshold) {
    if (params.imu.use) {
      LOG(INFO) << "large reprojection error (" << reprErr << "): run RANSAC";
      runRansac = true;
    }
    numInitIter += 2;
  }
  bool secondRansac = false;
  if(runRansac) {
    const bool ransacSuccess = runRansac3d2d(estimator, multiFrame->cameraSystem(), multiFrame,
                                             runRansac, ransacRemoveOutliers);
    T_WS1 = estimator.pose(StateId(currentFrameId));
    if (!ransacSuccess) {
      numInitIter += 4;
      secondRansac = true;
    }
  }

  // do optimisation
  std::vector<StateId> updatedStatesRealtime;
  if(!loopClosureLandmarksToUseExclusively && ctr > 3) {
    estimator.optimiseRealtimeGraph(
        numInitIter, updatedStatesRealtime, params.estimator.realtime_num_threads,
        false, true, isInitialized_);
    /*int numInliers = */removeOutliers<CAMERA_GEOMETRY>(estimator,
                                    params.nCameraSystem,
                                    estimator.multiFrame(StateId(currentFrameId)));
    estimator.optimiseRealtimeGraph(
      2, updatedStatesRealtime, params.estimator.realtime_num_threads,
      false, true, isInitialized_);
    T_WS1 = estimator.pose(StateId(currentFrameId));
  }
  if (ctr <= 3 && isInitialized_) {
    secondRansac = true;
  }

  // now the non-initialised ones
  for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im) {
    // the current frame to match
    const MultiFramePtr multiFrame = estimator.multiFrame(StateId(currentFrameId));
    const size_t numKeypoints = multiFrame->numKeypoints(im);
    if(numKeypoints == 0) {
      continue; // no points -- bad!
    }

    // prepare landmarks as visible in this frame
    AlignedMap<LandmarkId, LandmarkToMatch> landmarksToMatch;
    const kinematics::Transformation T_SC = *multiFrame->T_SC(im);
    const kinematics::Transformation T_WC1 = T_WS1 * T_SC;
    const kinematics::Transformation T_CW1 = T_WC1.inverse();

    // multithreaded matching
    const size_t num_matching_threads = size_t(params.frontend.num_matching_threads);

    std::vector<double> distances(numKeypoints,briskMatchingThreshold_);
    std::vector<LandmarkId> lmIds(numKeypoints);
    AlignedVector<Eigen::Vector4d> hps_W(numKeypoints, Eigen::Vector4d::Zero());
    std::vector<size_t> ctrs(num_matching_threads);

    std::vector<std::thread*> threads(num_matching_threads, nullptr);
    for(size_t t = 0; t<num_matching_threads; ++t) {
      threads[t] = new std::thread(
          &Frontend::matchToMapByThreadUnitialised<CAMERA_GEOMETRY>, this, t, num_matching_threads,
              std::cref(estimator), std::cref(params), currentFrameId,
              loopClosureLandmarksToUseExclusively, std::cref(T_WS1),
              std::cref(landmarksToMatchVec[im]), numKeypoints,
              std::cref(pointMap), im, std::cref(multiFrame), std::ref(distances),
              std::ref(lmIds), std::ref(hps_W), std::ref(ctrs));
    }

    for(size_t t = 0; t<num_matching_threads; ++t) {
      threads[t]->join();
      delete threads[t];
    }

    // now insert observations
    for(size_t k = 0; k < numKeypoints; ++k) {
      uint64_t previousId = multiFrame->landmarkId(im,k);
      if(lmIds[k].isInitialised()) {

        if(previousId && loopClosureLandmarksToUseExclusively) {
          // remove
          estimator.removeObservation(StateId(currentFrameId), im, k);
          oldIds.push_back(LandmarkId(previousId));
          newIds.push_back(lmIds[k]);
        }

        // check bad reprojections into existing frames
        MapPoint2 mpt;
        estimator.getLandmark(lmIds[k], mpt);
        if (hps_W[k].norm() > 1.0e-22) {
          bool badReprojections = false;
          for (const auto &obs : mpt.observations) {
            Eigen::Vector2d ptp;
            Eigen::Vector2d pt;
            const auto &mf = estimator.multiFrame(StateId(obs.frameId));
            mf->getKeypoint(obs.cameraIndex, obs.keypointIndex, pt);
            const auto &cam = mf->geometryAs<CAMERA_GEOMETRY>(obs.cameraIndex);
            const kinematics::Transformation T_WS = estimator.pose(StateId(obs.frameId));
            const kinematics::Transformation T_SC = estimator.extrinsics(StateId(obs.frameId),
                                                                         obs.cameraIndex);
            //Eigen::Vector4d hpW = mpt.point;
            Eigen::Vector4d hpC = T_SC.inverse() * T_WS.inverse() * hps_W[k];
            auto s = cam->projectHomogeneous(hpC, &ptp);
            if (!(s == cameras::ProjectionStatus::Successful && (pt - ptp).norm() < 4.0)) {
              badReprojections = true;
              break;
            }
          }
          if (badReprojections) {
            continue;
          }
        }

        Eigen::Vector2d pt1;
        Eigen::Vector2d pt1p;
        multiFrame->getKeypoint(im, k, pt1);
        const auto &cam1 = multiFrame->geometryAs<CAMERA_GEOMETRY>(im);
        if (hps_W[k].norm() > 1.0e-22 && !estimator.isLandmarkInitialised(lmIds[k])) { //ugly
          // check current reprojection
          auto s1 = cam1->projectHomogeneous(T_WC1.inverse() * hps_W[k], &pt1p);
          if (!(s1 == cameras::ProjectionStatus::Successful && (pt1 - pt1p).norm() < 4.0)) {
            continue;
          }

          // accept and set position
          estimator.setLandmark(lmIds[k], hps_W[k], true);
        } else {
          auto s1 = cam1->projectHomogeneous(T_WC1.inverse() * Eigen::Vector4d(mpt.point), &pt1p);
          if (!(s1 == cameras::ProjectionStatus::Successful && (pt1 - pt1p).norm() < 4.0)) {
            continue;
          }
        }

        // accept and set observation
        multiFrame->setLandmarkId(im, k, lmIds[k].value());
        estimator.addObservation<CAMERA_GEOMETRY>(
            lmIds[k], StateId(currentFrameId), im, k);
        if (landmarksToMatch[lmIds[k]].ignore) {
          estimator.setObservationInformation(StateId(currentFrameId), im, k,
                                              Eigen::Matrix2d::Identity()*0.00001);
        }
        ctr++;
      }
    }
  }
  //OKVIS_ASSERT_TRUE(Exception, estimator.areLandmarksInFrontOfCameras(), "after non-initialised match to map")

  // merge landmarks, if loop-closure matching
  if(loopClosureLandmarksToUseExclusively) {
    estimator.mergeLandmarks(oldIds, newIds);
  }

  // final two steps optimisation
  if (secondRansac) {
    LOG(INFO) << "Running RANSAC also with uninitialised landmarks";
    const bool ransacSuccess = runRansac3d2d(estimator, multiFrame->cameraSystem(), multiFrame,
                                             secondRansac, ransacRemoveOutliers);
    T_WS1 = estimator.pose(StateId(currentFrameId));
    if (!ransacSuccess) {
      numInitIter += 4;
    }
    estimator.optimiseRealtimeGraph(
    numInitIter, updatedStatesRealtime, params.estimator.realtime_num_threads,
        false, true, isInitialized_);
  }
  //OKVIS_ASSERT_TRUE(Exception, estimator.areLandmarksInFrontOfCameras(), "after match to map")

  return ctr;
}

// Match a new multiframe to existing keyframes:
template <class CAMERA_GEOMETRY>
void Frontend::matchToMapByThread(
    size_t threadIdx, size_t numThreads, const Estimator &estimator,
    const okvis::ViParameters& params, const uint64_t currentFrameId,
    const std::set<LandmarkId>* loopClosureLandmarksToUseExclusively,
    const kinematics::Transformation& T_WS1,
    const AlignedMap<LandmarkId, LandmarkToMatch>& landmarksToMatch,
    size_t numKeypoints, const MapPoints& pointMap,
    size_t im, const MultiFramePtr&  multiFrame, std::vector<double>& distances,
    std::vector<LandmarkId>& lmIds, AlignedVector<Eigen::Vector4d>& hps_W,
    std::vector<size_t>& ctrs,
    std::vector<double>& reprErrors) const {

  const kinematics::Transformation T_SC = *multiFrame->T_SC(im);
  const kinematics::Transformation T_WC1 = T_WS1 * T_SC;
  const kinematics::Transformation T_CW1 = T_WC1.inverse();

  const double f = 0.5*(multiFrame->geometryAs<CAMERA_GEOMETRY>(im)->focalLengthU()
                   + multiFrame->geometryAs<CAMERA_GEOMETRY>(im)->focalLengthV());

  const double reprojectionThreshold = params.imu.use ? 3.0+f*0.06 : 3.0+f*0.34;
  const double reprojectionThresholdSq = reprojectionThreshold * reprojectionThreshold;

  ctrs[threadIdx] = 0;

  // go through all landmarks
  const size_t segment = numKeypoints/numThreads;
  const size_t startK = segment*threadIdx;
  const size_t endK = threadIdx+1 == numThreads ? numKeypoints : startK + segment;
  const uchar* ddata = multiFrame->keypointDescriptor(im, 0);
  const size_t descBytes = descriptorBytes();
  Eigen::Matrix2Xd keypoints(2,numKeypoints);
  std::vector<bool> use(numKeypoints, true);
  for(size_t k = startK; k < endK; k++) {
    Eigen::Vector2d keypoint;
    multiFrame->getKeypoint(im, k, keypoint);
    keypoints.col(k) = keypoint;
    const uint64_t previousId = multiFrame->landmarkId(im,k);
    if(previousId&&!loopClosureLandmarksToUseExclusively) {
      use[k] = false; // I don't remember why this could happen -- just being paranoid.
      continue; // already matched
    }
  }
  for(auto it = landmarksToMatch.begin(); it != landmarksToMatch.end(); ++it) {

    if(!it->second.is3d) {
      continue;
    }

    if(loopClosureLandmarksToUseExclusively) {
      if(!loopClosureLandmarksToUseExclusively->count(it->first)) {
        continue; // skip non-loop-closure points in this case
      }
    }

    // match all present descriptors
    const Eigen::Vector2d projection = it->second.projection;
    for(size_t k = startK; k < endK; k++) {

      if(!use[k]) {
        continue;
      }

      // also check image distance, unless tracking lost.
      const Eigen::Vector2d reprDist = projection - keypoints.col(k);
      if (reprDist.dot(reprDist) > reprojectionThresholdSq) {
        continue;
      }

      const uchar* descriptorK = ddata + k*descBytes;
      for(int d = 0; d<it->second.descriptors.rows; ++d) {
        const double dist = descriptorDist(
            descriptorK,
            it->second.descriptors.data + d*descBytes);
        if(dist < distances[k]) {
          distances[k] = dist;
          lmIds[k] = it->first;
          ctrs[threadIdx]++;
          reprErrors[threadIdx] += sqrt(reprDist.dot(reprDist));
        }
      }
    }
  }
  reprErrors[threadIdx] /= double(ctrs[threadIdx]);
}

// Match a new multiframe to existing keyframes:
template <class CAMERA_GEOMETRY>
void Frontend::matchToMapByThreadUnitialised(
    size_t threadIdx, size_t numThreads, const Estimator &estimator,
    const okvis::ViParameters& params, const uint64_t currentFrameId,
    const std::set<LandmarkId>* loopClosureLandmarksToUseExclusively,
    const kinematics::Transformation& T_WS1,
    const AlignedMap<LandmarkId, LandmarkToMatch>& landmarksToMatch,
    size_t numKeypoints, const MapPoints& pointMap,
    size_t im, const MultiFramePtr&  multiFrame, std::vector<double>& distances,
    std::vector<LandmarkId>& lmIds, AlignedVector<Eigen::Vector4d>& hps_W,
    std::vector<size_t>& ctrs) const {

  const kinematics::Transformation T_SC = *multiFrame->T_SC(im);
  const kinematics::Transformation T_WC1 = T_WS1 * T_SC;
  const kinematics::Transformation T_CW1 = T_WC1.inverse();

  ctrs[threadIdx] = 0;

  const double focalLength =
      0.5*(multiFrame->geometryAs<CAMERA_GEOMETRY>(im)->focalLengthU()
      + multiFrame->geometryAs<CAMERA_GEOMETRY>(im)->focalLengthV());

  // go through all landmarks
  const size_t segment = numKeypoints/numThreads;
  const size_t startK = segment*threadIdx;
  const size_t endK = threadIdx+1 == numThreads ? numKeypoints : startK + segment;
  const uchar* ddata = multiFrame->keypointDescriptor(im, 0);
  const size_t descBytes = descriptorBytes();
  Eigen::Matrix3Xd e_Ws(3,numKeypoints);
  std::vector<uint64_t> previousIds(numKeypoints,0);
  std::vector<bool> use(numKeypoints,false);
  for(size_t k = startK; k < endK; k++) {
    Eigen::Vector3d e1_C;
    if(multiFrame->getBackProjection(im, k, e1_C)){
      const Eigen::Vector3d e1_W = T_WC1.C()*e1_C.normalized();
      e_Ws.col(k) = e1_W;
      const uint64_t previousId = multiFrame->landmarkId(im,k);
      previousIds[k] = previousId;
      if(previousId&&!loopClosureLandmarksToUseExclusively) {
        continue; // already matched
      }
      use[k] = true;
    }
  }
  const double sigma = 1.0/focalLength;
  const double cos6Sigma = cos(6.0*sigma);
  for(auto it = landmarksToMatch.begin(); it != landmarksToMatch.end(); ++it) {

    if(it->second.is3d) {
      continue;
    }

    if(loopClosureLandmarksToUseExclusively) {
      if(!loopClosureLandmarksToUseExclusively->count(it->first)) {
        continue; // skip non-loop-closure points in this case
      }
    }

    // match all present descriptors
    for(size_t k = startK; k < endK; k++) {

      if(!use[k]) {
        continue;
      }

      // also check epipolar distance (later)
      const Eigen::Vector3d e1_W=e_Ws.col(k);
      const uchar* descriptorK = ddata + k*descBytes;
      for(int d = 0; d<it->second.descriptors.rows; ++d) {
        const double dist = descriptorDist(
            descriptorK, it->second.descriptors.data + d*descBytes);

        if(dist < distances[k]) {

          // epipolar distance check
          const Eigen::Vector3d e0_W = it->second.e_W.col(d);
          const Eigen::Vector3d r0_W = it->second.r_W.col(d);

          if(e0_W.dot(e1_W)<cos6Sigma) { // otherwise parallel... will be OK.
            const Eigen::Vector3d et_W = (T_WC1.r() - r0_W).normalized();
            const Eigen::Vector3d n0_W = e0_W.cross(et_W).normalized();
            const Eigen::Vector3d n1_W = e1_W.cross(et_W).normalized();
            if((n0_W.dot(n1_W) < cos6Sigma)) {
              continue; // not in epipolar plane
            }
            if((e0_W.cross(e1_W)).dot((n0_W + n0_W).normalized())>0.0) {
              continue; // divergent rays
            }
          }

          // try triangulation
          bool isValid = false;
          bool isParallel = false;
          Eigen::Vector4d hp_W = triangulation::triangulateFast(
              r0_W, e0_W, T_WC1.r(), e1_W, sigma, isValid, isParallel);

          if(!isValid) {
            continue;
          }

          // check if too close (out of focus)
          const Eigen::Vector3d p_W = hp_W.head<3>()/hp_W[3];
          if((p_W-r0_W).norm() < 0.2) {
            isValid = false;
          }
          if((p_W-T_WC1.r()).norm() < 0.2) {
            isValid = false;
          }
          if(!isValid) {
            continue;
          }

          if(it->first.value()==previousIds[k]) {
            ctrs[threadIdx]++; // still counts, already correct match...
            break; // the match is already done...
          }

          distances[k] = dist;
          lmIds[k] = it->first;
          ctrs[threadIdx]++;
          if(!isParallel) {
            hps_W[k] = hp_W;
          }
        }
      }
    }
  }
}

/// \brief Temporary match info storage.
struct MatchInfo {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector4d hp_W; ///< 3D point.
  size_t k1 = 0; ///< Match idx.
  bool matching = false; ///< Does it match?
  bool initialisable = false; ///< Initialisable?
  double quality = 0.0; ///< 3D quality.
};

template <class CAMERA_GEOMETRY>
int Frontend::matchMotionStereo(Estimator& estimator, const ViParameters &params,
                                 const uint64_t currentFrameId, bool& rotationOnly) {
  int retCtr = 0;
  rotationOnly = true;

  kinematics::Transformation T_WS1 = estimator.pose(StateId(currentFrameId));

  // find close frames
  TimerSwitchable matchMotionStereoTimer2("2.02.1 match motion stereo: prepare");
  std::set<StateId> allFrames;
  allFrames.insert(estimator.keyFrames().begin(), estimator.keyFrames().end());
  allFrames.insert(estimator.imuFrames().begin(), estimator.imuFrames().end());
  for(const auto & id : estimator.imuFrames()) {
    if(!estimator.isKeyframe(id)) {
      allFrames.erase(id);
    }
  }
  StateId previousFrameId = estimator.stateIdByAge(1);
  std::vector<std::pair<double, StateId>> overlaps;
  for(auto & id : allFrames) {
    if(id == previousFrameId) {
      overlaps.push_back(std::pair<double, StateId>(1.0, id));
      continue;
    }
    const double overlap = estimator.overlapFraction(
          estimator.multiFrame(previousFrameId),estimator.multiFrame(id));
    overlaps.push_back(std::pair<double, StateId>(overlap, id));
  }
  std::sort(overlaps.begin(), overlaps.end());
  std::vector<StateId> matchFrameIds;

  for(size_t i=0; i<overlaps.size(); ++i) {
    if(overlaps[overlaps.size()-i-1].first <= 1.0e-8) break;
    matchFrameIds.push_back(overlaps[overlaps.size()-i-1].second);
  }

  matchMotionStereoTimer2.stop();

  kinematics::Transformation T_WS0;
  bool firstFrame = true;
  size_t overlapRank = 0;  // matchFrameIds is sorted best-overlap-first
  for (auto olderFrameId : matchFrameIds) {
    const size_t frameRank = overlapRank++;
    T_WS0 = estimator.pose(olderFrameId);
    for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im) {
      const kinematics::Transformation T_SC0 = estimator.extrinsics(StateId(olderFrameId), im);
      const kinematics::Transformation T_SC1 = estimator.extrinsics(StateId(currentFrameId), im);
      const kinematics::Transformation T_WC0 = T_WS0 * T_SC0;
      const kinematics::Transformation T_WC1 = T_WS1 * T_SC1;
      // match
      MultiFramePtr multiFrame0 = estimator.multiFrame(olderFrameId);
      MultiFramePtr multiFrame1 = estimator.multiFrame(StateId(currentFrameId));
      const size_t k0Size = multiFrame0->numKeypoints(im);
      const size_t k1Size = multiFrame1->numKeypoints(im);
      const auto camera = multiFrame0->geometryAs<CAMERA_GEOMETRY>(im);
      const double f0 = 0.5* (camera->focalLengthU() + camera->focalLengthV());

      // preprocess matchable set, get descriptors close in memory
      const size_t descBytes = descriptorBytes();
      std::vector<size_t> k1s;
      k1s.reserve(k1Size);
      cv::Mat desc1(int(k1Size), int(descBytes), CV_8UC1);
      for(size_t k1 = 0; k1 < k1Size; ++k1) {
        const uint64_t id1 = multiFrame1->landmarkId(im, k1);
        if(id1) {
          continue; // already matched
        }
        std::memcpy(
            desc1.data+descBytes*k1s.size(),
            multiFrame1->keypointDescriptor(im, k1), descBytes);
        k1s.push_back(k1);
      }
      desc1 = desc1(cv::Rect(0,0,int(descBytes),k1s.size()));

      // LighterGlue proposals for the best-overlap frames (ADR-0040 stage C):
      // the top motion_stereo_top_n of the overlap-sorted matchFrameIds use
      // LighterGlue (one candidate per k0), the rest brute-force distance.
      std::vector<int> lgMatch;  // k0 (older frame) -> k1 (current) or -1
      std::vector<int> k1ToKk;   // original k1 -> compacted kk index or -1
      bool lgMode = false;
      if (frameRank < size_t(std::max(0, xfeatParams_.motion_stereo_top_n))) {
        lgMode = lighterGluePairProposals(*multiFrame0, im, *multiFrame1, im,
                                          lgMatch);
      }
      if (lgMode) {
        k1ToKk.assign(k1Size, -1);
        for (size_t kk = 0; kk < k1s.size(); ++kk) {
          k1ToKk[k1s[kk]] = int(kk);
        }
      }

      AlignedVector<MatchInfo> matchInfos(k0Size);

      // vector container stores threads
      std::vector<std::thread> workers;
      for (size_t t = 0; t < size_t(params.frontend.num_matching_threads); t++) {
        workers.push_back(std::thread([this, t, k0Size, im, &multiFrame0, &estimator, f0, k1s,
                                      &T_WC0, &T_WC1, &multiFrame1, &olderFrameId, &matchInfos,
                                       &params, &camera, desc1, descBytes,
                                       lgMode, &lgMatch, &k1ToKk]() {
          for(size_t k0 = t; k0 < k0Size; k0 += size_t(params.frontend.num_matching_threads)) {
            uint64_t id0 = multiFrame0->landmarkId(im, k0);
            if(id0) {
              if(!estimator.isLandmarkAdded(LandmarkId(id0))) {
                continue; // weird
              }
              if(estimator.isLandmarkInitialised(LandmarkId(id0))) {
                continue; // already matched
              }
            }

            double distances = briskMatchingThreshold_;
            bool initialisable = false;
            double quality = 0.0;
            Eigen::Vector4d hps_W(0,0,0,0);
            size_t k1_max=1000;

            // pre-fetch frame 0 stuff
            const uchar* d0 = multiFrame0->keypointDescriptor(im, k0);
            double size0;
            multiFrame0->getKeypointSize(im, k0, size0);
            Eigen::Vector2d pt0;
            multiFrame0->getKeypoint(im, k0, pt0);
            Eigen::Vector3d e0_C;
            if(!multiFrame0->getBackProjection(im, k0, e0_C)) continue;
            const Eigen::Vector3d e0_W = (T_WC0.C()*e0_C).normalized();
            const double sigma = size0/f0 * 0.125;

            if(estimator.isObserved(KeypointIdentifier{olderFrameId.value(), im, k0})) {
              continue; // already matched
            }

            size_t kkFrom = 0, kkTo = k1s.size();
            if (lgMode) {
              const int k1p = lgMatch[k0];
              if (k1p < 0 || k1ToKk[size_t(k1p)] < 0) {
                continue;  // no proposal, or proposed k1 already matched
              }
              kkFrom = size_t(k1ToKk[size_t(k1p)]);
              kkTo = kkFrom + 1;
            }
            for(size_t kk = kkFrom; kk < kkTo; ++kk) {
              const size_t k1 = k1s[kk];
              // LighterGlue-proposed pair: distance 0 accepts, still subject
              // to the triangulation checks below.
              const double dist =
                  lgMode ? 0.0 : descriptorDist(d0, desc1.data+kk*descBytes);
              if(dist < distances) {
                // it's a match!

                // triangulate
                bool isValid = false;
                bool isParallel = false;
                Eigen::Vector3d e1_C;
                if(!multiFrame1->getBackProjection(im, k1, e1_C)) continue;
                const Eigen::Vector3d e1_W = (T_WC1.C()*e1_C).normalized();
                if(e0_W.dot(e1_W) < 0.5) continue;

                Eigen::Vector4d hp_W = triangulation::triangulateFast(
                      T_WC0.r(), e0_W, T_WC1.r(), e1_W, sigma, isValid, isParallel);

                if(!isValid) {
                  continue;
                }

                // check if too close (out of focus)
                const Eigen::Vector4d hp_C0 = (T_WC0.inverse()*hp_W);
                const Eigen::Vector4d hp_C1 = (T_WC1.inverse()*hp_W);

                if(e0_W.transpose()*e1_W < 0.8) {
                  isValid = false;
                }

                hp_W = hp_W/hp_W[3];
                if(hp_C0[2]/hp_C0[3] < 0.2) {
                  isValid = false;
                }
                if(hp_C1[2]/hp_C1[3] < 0.2) {
                  isValid = false;
                }

                // remember
                if(/*dist<distances && */isValid) {
                  k1_max = k1;
                  distances=dist;
                  quality = acos((hp_W.head<3>()-T_WC0.r()).normalized()
                                 .dot((hp_W.head<3>()-T_WC1.r()).normalized()));
                  hps_W = hp_W;
                  initialisable = !isParallel;
                }
              }
            }

            // add observations and initialise
            if(distances < briskMatchingThreshold_) {
              Eigen::Vector2d pt1p;
              Eigen::Vector2d pt1;
              multiFrame1->getKeypoint(im, k1_max, pt1);
              auto s1 = camera->projectHomogeneous(T_WC1.inverse()*hps_W, &pt1p);
              if(s1 == cameras::ProjectionStatus::Successful && (pt1-pt1p).norm()<4.0) {
                matchInfos[k0] = MatchInfo{hps_W, k1_max, true, initialisable, quality};
              }
            }
          }
        }));
      }

      // join all matcher threads
      std::for_each(workers.begin(), workers.end(), [](std::thread &worker) {
          worker.join();
      });

      // finally insert the actual matches
      for(size_t k0=0; k0<k0Size; ++k0) {
        const MatchInfo & mInfo = matchInfos.at(k0);
        uint64_t id0 = multiFrame0->landmarkId(im, k0);
        if(!mInfo.matching) {
          continue;
        }

        if(id0) {
          if(estimator.isLandmarkInitialised(LandmarkId(id0))) {
            continue; // already matched
          }
          if(!estimator.isLandmarkAdded(LandmarkId(id0))) {
            continue; // weird!
          }
        }
        if(estimator.isObserved(KeypointIdentifier{olderFrameId.value(), im, k0})) {
          continue; // already matched
        }

        const uint64_t id1 = multiFrame1->landmarkId(im, mInfo.k1);
        if(id1) {
          continue; // already matched
        }

        if(id0){
          MapPoint2 lm;
          estimator.getLandmark(LandmarkId(id0), lm);
          if(lm.quality<mInfo.quality) {
            estimator.setLandmark(LandmarkId(id0), mInfo.hp_W, mInfo.initialisable);
          }
        } else {
          id0 = estimator.addLandmark(mInfo.hp_W, mInfo.initialisable).value();
          multiFrame0->setLandmarkId(im, k0, id0);
          OKVIS_ASSERT_TRUE_DBG(Exception, estimator.isLandmarkAdded(LandmarkId(id0)),
                              id0<<" not added, bug")
          estimator.addObservation<CAMERA_GEOMETRY>(LandmarkId(id0), StateId(olderFrameId), im, k0);
        }

        multiFrame1->setLandmarkId(im, mInfo.k1, id0);
        estimator.addObservation<CAMERA_GEOMETRY>(
              LandmarkId(id0), StateId(currentFrameId), im, mInfo.k1);
        retCtr++;
      }
    }

    bool rotationOnly_tmp = false;
    static const bool removeOutliers = true;

    // do RANSAC 2D2D for initialization only
    const bool initialisePose =  (!isInitialized_);
    if(!isInitialized_) {
      runRansac2d2d(estimator, params, currentFrameId, olderFrameId.value(), initialisePose,
                    removeOutliers, rotationOnly_tmp);
    }

    if (firstFrame) {
      rotationOnly = rotationOnly_tmp;
      firstFrame = false;
    }
  }

  return retCtr;
}

// Match the frames inside the multiframe to each other to initialise new landmarks.
template <class CAMERA_GEOMETRY>
void Frontend::matchStereo(Estimator &estimator, std::shared_ptr<okvis::MultiFrame> multiFrame,
                           const okvis::ViParameters& params, bool asKeyframe) {
  const size_t camNumber = multiFrame->numFrames();
  const uint64_t mfId = multiFrame->id();

  // needed later:
  kinematics::Transformation T_WS = estimator.pose(StateId(mfId));

  for (size_t im0 = 0; im0 < camNumber; im0++) {
    const kinematics::Transformation T_SC0 = *multiFrame->T_SC(im0);

    for (size_t im1 = im0 + 1; im1 < camNumber; im1++) {
      // first, check the possibility for overlap
      // FIXME: implement this in the Multiframe...!!

      // check overlap
      if (!multiFrame->hasOverlap(im0, im1)) {
        continue;
      }

      // useful later:
      const kinematics::Transformation T_SC1 = *multiFrame->T_SC(im1);
      const kinematics::Transformation T_WC0 = T_WS * T_SC0;
      const kinematics::Transformation T_WC1 = T_WS * T_SC1;

      {
        // match
        MultiFramePtr multiFrame = estimator.multiFrame(StateId(mfId));
        const size_t k0Size = multiFrame->numKeypoints(im0);
        const size_t k1Size = multiFrame->numKeypoints(im1);
        const auto camera0 = multiFrame->geometryAs<CAMERA_GEOMETRY>(im0);
        const auto camera1 = multiFrame->geometryAs<CAMERA_GEOMETRY>(im1);
        const double f0 = 0.5* (camera0->focalLengthU() + camera0->focalLengthV());
        const double f1 = 0.5* (camera1->focalLengthU() + camera1->focalLengthV());
        // Pass 0: brute-force (mutual-NN on the float path). Pass 1, only
        // when pass 0 found fewer than xfeat.stereo_min_nn_matches and a
        // LighterGlue engine is loaded: one proposal per still-unmatched k0
        // (KB 02 decision table — LighterGlue is the stereo FALLBACK, T-0114;
        // ADR-0040 stage B). The triangulation validation is shared.
        std::vector<int> lgMatch;
        bool lgMode = false;
        uint64_t numStereoMatches = 0;
        std::vector<bool> matched0(k0Size, false), matched1(k1Size, false);
        for(int pass = 0; pass < 2; ++pass) {
        if(pass == 1) {
          if(!floatDescriptors() ||
             numStereoMatches >= uint64_t(std::max(0, xfeatParams_.stereo_min_nn_matches)) ||
             !lighterGluePairProposals(*multiFrame, im0, *multiFrame, im1, lgMatch)) {
            break;
          }
          lgMode = true;
          std::lock_guard<std::mutex> lock(statsMutex_);
          ++stats_.stereoLgFallbacks;
        }
        for(size_t k0 = 0; k0 < k0Size; ++k0) {
          if(lgMode && (matched0[k0] || (lgMatch[k0] >= 0 && matched1[size_t(lgMatch[k0])]))) {
            continue;  // keep pass-0 matches; never re-add or steal them
          }

          double distances = briskMatchingThreshold_;
          bool initialisable=  false;
          Eigen::Vector4d hps_W;
          size_t k1_match = 0;

          size_t k1From = 0, k1To = k1Size;
          if (lgMode) {
            if (lgMatch[k0] < 0) {
              continue;  // LighterGlue proposes nothing for this keypoint
            }
            k1From = size_t(lgMatch[k0]);
            k1To = k1From + 1;
          }
          for(size_t k1 = k1From; k1 < k1To; ++k1) {
            // LighterGlue already decided the correspondence: distance 0
            // accepts it, still subject to the triangulation checks below.
            const double dist = lgMode ? 0.0 : descriptorDist(
                multiFrame->keypointDescriptor(im0, k0),
                  multiFrame->keypointDescriptor(im1, k1));
            if(dist < distances) {
              // it's a match!
              double size0, size1;
              multiFrame->getKeypointSize(im0, k0, size0);
              multiFrame->getKeypointSize(im1, k1, size1);
              Eigen::Vector2d pt0, pt1;
              multiFrame->getKeypoint(im0, k0, pt0);
              multiFrame->getKeypoint(im1, k1, pt1);
              const double sigma = std::max(size0/f0, size1/f1) * 0.125;

              // triangulate
              bool isValid = false;
              bool isParallel = false;
              Eigen::Vector3d e0_C, e1_C;
              if(!multiFrame->getBackProjection(im0, k0, e0_C)) continue;
              if(!multiFrame->getBackProjection(im1, k1, e1_C)) continue;
              Eigen::Vector3d e0_W = (T_WC0.C()*e0_C).normalized();
              Eigen::Vector3d e1_W = (T_WC1.C()*e1_C).normalized();
              Eigen::Vector4d hp_W = triangulation::triangulateFast(
                    T_WC0.r(), e0_W, T_WC1.r(), e1_W, sigma, isValid, isParallel);

              // check if too close
              const Eigen::Vector4d hp_C0 = (T_WC0.inverse()*hp_W);
              const Eigen::Vector4d hp_C1 = (T_WC1.inverse()*hp_W);

              hp_W = hp_W/hp_W[3];

              if(hp_C0[2]/hp_C0[3] < 0.1) {
                isValid = false;
              }
              if(hp_C1[2]/hp_C1[3] <  0.1) {
                isValid = false;
              }

              if(e0_W.transpose()*e1_W < 0.8) {
                isValid = false;
              }

              // add observations and initialise
              if(isValid) {
                distances = dist;
                hps_W = hp_W;
                k1_match = k1;
                initialisable = !isParallel;
              }
            }
          }

          if(floatDescriptors() && !lgMode && distances<briskMatchingThreshold_) {
            // Mutual-NN on the float path (T-0113, KB 02 stereo row): k0 must
            // also be the nearest neighbour of k1_match among all k0.
            const unsigned char* d1 = multiFrame->keypointDescriptor(im1, k1_match);
            for(size_t k = 0; k < k0Size; ++k) {
              if(k != k0 && descriptorDist(multiFrame->keypointDescriptor(im0, k), d1) < distances) {
                distances = briskMatchingThreshold_; // reject: not mutual
                break;
              }
            }
          }

          if(distances<briskMatchingThreshold_) {
            ++numStereoMatches;
            matched0[k0] = true;
            matched1[k1_match] = true;
            Eigen::Vector2d pt0, pt1;
            multiFrame->getKeypoint(im0, k0, pt0);
            multiFrame->getKeypoint(im1, k1_match, pt1);
            uint64_t lmId = 0;
            const uint64_t id0 = multiFrame->landmarkId(im0, k0); // may change!!
            uint64_t id1 = multiFrame->landmarkId(im1, k1_match);
            bool add0 = false;
            bool add1 = false;
            if(id0 && id1) {
              if (id0 != id1) {
                estimator.mergeLandmark(LandmarkId(id1), LandmarkId(id0));
                id1 = id0;
              }
              if(!estimator.isLandmarkInitialised(LandmarkId(id0))) {
                // only re-assess initialisation
                if(initialisable) {
                  //estimator.setLandmarkInitialized(id0, initialiseable);
                  estimator.setLandmark(LandmarkId(id0), hps_W, true); /// \todo check true
                }
              } // else we do nothing, because already initialised and matched.
            } else if(id1) {
              // only add observation into frame0
              lmId = id1;
              add0 = true;

            } else if(id0) {
              // only add observation into frame1
              lmId = id0;
              add1 = true;
            } else {
              if(!asKeyframe){
                continue; // we don't want to create new stuff from non-keyframes
              }
              add0 = true;
              add1 = true;
              // need new point

              lmId = estimator.addLandmark(hps_W, initialisable).value();
              OKVIS_ASSERT_TRUE_DBG(
                  Exception, estimator.isLandmarkAdded(LandmarkId(lmId)),
                  lmId<<" not added, bug")
            }
            if(add0) {
              // verify (again, because landmark may not have been reset)
              Eigen::Vector2d pt0p;
              MapPoint2 mapPoint;
              estimator.getLandmark(LandmarkId(lmId), mapPoint);
              Eigen::Vector4d hp_eff_W = mapPoint.point;
              auto s0 = camera0->projectHomogeneous(T_WC0.inverse()*hp_eff_W, &pt0p);
              if(s0 == cameras::ProjectionStatus::Successful && (pt0-pt0p).norm()<4.0) {
                // safe to add.
                multiFrame->setLandmarkId(im0, k0, lmId);
                estimator.addObservation<CAMERA_GEOMETRY>(LandmarkId(lmId), StateId(mfId), im0, k0);
              }
            }
            if(add1) {
              // verify (again, because landmark may not have been reset)
              Eigen::Vector2d pt1p;
              MapPoint2 mapPoint;
              estimator.getLandmark(LandmarkId(lmId), mapPoint);
              Eigen::Vector4d hp_eff_W = mapPoint.point;
              auto s1 = camera1->projectHomogeneous(T_WC1.inverse()*hp_eff_W, &pt1p);
              if(s1 == cameras::ProjectionStatus::Successful && (pt1-pt1p).norm()<4.0) {
                multiFrame->setLandmarkId(im1, k1_match, lmId);
                estimator.addObservation<CAMERA_GEOMETRY>(
                  LandmarkId(lmId), StateId(mfId), im1, k1_match);
              }
            }
          }
        }
        }  // pass
        {
          std::lock_guard<std::mutex> lock(statsMutex_);
          stats_.stereoMatches += numStereoMatches;
          ++stats_.stereoCalls;
        }
      }
    }
  }

  // TODO: for more than 2 cameras check that there were no duplications!

  // TODO: ensure 1-1 matching (done for the float path above).
}
template<class CAMERA_GEOMETRY>
int Frontend::removeOutliers(Estimator &estimator,
                             const okvis::cameras::NCameraSystem &nCameraSystem,
                             std::shared_ptr<okvis::MultiFrame> currentFrame)
{
  const size_t camNumber = currentFrame->numFrames();
  const uint64_t mfId = currentFrame->id();

  // needed later:
  kinematics::Transformation T_WS = estimator.pose(StateId(mfId));

  int ctr = 0;

  for (size_t im = 0; im < camNumber; im++) {
    const kinematics::Transformation T_SC = *currentFrame->T_SC(im);
    const kinematics::Transformation T_WCi = T_WS * T_SC;
    const kinematics::Transformation T_CiW = T_WCi.inverse();
    const size_t kSize = currentFrame->numKeypoints(im);
    for (size_t k = 0; k < kSize; ++k) {
      uint64_t lmId = currentFrame->landmarkId(im, k);
      if (lmId) {
        Eigen::Vector2d pt, proj;
        if (currentFrame->getKeypoint(im, k, pt)) {
          //Eigen::Vector3d e_Ci;
          //if (currentFrame->getBackProjection(im, k, e_Ci)) {
          MapPoint2 lm;
          if (estimator.getLandmark(LandmarkId(lmId), lm)) {
            bool remove = false;
            const Eigen::Vector4d hp_W = lm.point;
            Eigen::Vector4d hp_Ci = T_CiW * hp_W;
            const auto camera = currentFrame->geometryAs<CAMERA_GEOMETRY>(im);
            if (cameras::ProjectionStatus::Successful == camera->projectHomogeneous(hp_Ci, &proj)) {
              if ((proj - pt).norm() > 4.0) {
                remove = true;
              }
            } else {
              remove = true;
            }
            if (!remove) {
              ctr++;
            } else {
              estimator.removeObservation(StateId(mfId), im, k);
            }
          }
        }
      }
    }
  }
  return ctr;
}

// Perform 3D/2D RANSAC.
bool Frontend::runRansac3d2d(
    Estimator &estimator, const okvis::cameras::NCameraSystem& nCameraSystem,
    std::shared_ptr<okvis::MultiFrame> currentFrame, bool initializePose, bool removeOutliers) {
  if (estimator.numFrames() < 2) {
    // nothing to match against, we are just starting up.
    return false;
  }

  /////////////////////
  //   KNEIP RANSAC
  /////////////////////
  int numInliers = 0;

  // absolute pose adapter for Kneip toolchain
  opengv::absolute_pose::FrameNoncentralAbsoluteAdapter adapter(
    estimator, nCameraSystem, currentFrame);

  size_t numCorrespondences = adapter.getNumberCorrespondences();
  if (numCorrespondences < 10) return int(numCorrespondences);

  // create a RelativePoseSac problem and RANSAC
  typedef opengv::sac_problems::absolute_pose::FrameAbsolutePoseSacProblem<
        opengv::absolute_pose::FrameNoncentralAbsoluteAdapter> AbsoluteModel;
  opengv::sac::Ransac<AbsoluteModel> ransac;
  std::shared_ptr<AbsoluteModel> absposeproblem_ptr(
        new AbsoluteModel(adapter, AbsoluteModel::Algorithm::GP3P));
  ransac.sac_model_ = absposeproblem_ptr;
  ransac.threshold_ = 16;
  ransac.max_iterations_ = 50;
  // initial guess not needed...
  // run the ransac
  ransac.computeModel(0);

  // deal with outliers and assign transformation
  numInliers = int(ransac.inliers_.size());
  if (numInliers >= 10 && double(ransac.inliers_.size())/double(numCorrespondences)>0.7) {
    // kick out outliers:
    if(removeOutliers) {
      std::vector<bool> inliers(numCorrespondences, false);
      for (size_t k = 0; k < ransac.inliers_.size(); ++k) {
        inliers.at(size_t(ransac.inliers_.at(k))) = true;
      }

      for (size_t k = 0; k < numCorrespondences; ++k) {
        if (!inliers[k]) {
          // get the landmark id:
          size_t camIdx = size_t(adapter.camIndex(k));
          size_t keypointIdx = size_t(adapter.keypointIndex(k));

          // remove observation
          estimator.removeObservation(StateId(currentFrame->id()), camIdx, keypointIdx);
        }
      }
    }

    // assign transformation
    Eigen::Matrix4d T_WS_mat = Eigen::Matrix4d::Identity();
    T_WS_mat.topLeftCorner<3, 4>() = ransac.model_coefficients_;
    kinematics::Transformation T_WS = kinematics::Transformation(T_WS_mat);
    if(initializePose) {
      estimator.setPose(StateId(currentFrame->id()), T_WS);
    }
    return true;
  } else {
    LOG(INFO) << "RANSAC FAIL: " << numInliers << " inliers, ratio = "
              << double(ransac.inliers_.size())/double(numCorrespondences);
  }
  return false;
}

// Perform 2D/2D RANSAC.
int Frontend::runRansac2d2d(Estimator &estimator, const okvis::ViParameters& params,
                            uint64_t currentFrameId, uint64_t olderFrameId,
                            bool initializePose, bool removeOutliers, bool& rotationOnly) {
  // match 2d2d
  rotationOnly = false;
  const size_t numCameras = params.nCameraSystem.numCameras();

  int totalInlierNumber = 0;
  bool rotation_only_success = false;
  bool rel_pose_success = false;

  // run relative RANSAC
  for (size_t im = 0; im < numCameras; ++im) {
    // relative pose adapter for Kneip toolchain
    opengv::relative_pose::FrameRelativeAdapter adapter(estimator, params.nCameraSystem,
                                                        olderFrameId, im, currentFrameId, im);

    size_t numCorrespondences = adapter.getNumberCorrespondences();

    if (numCorrespondences < 10)
      continue;  // won't generate meaningful results. let's hope the few corresp. are inliers!!

    // try both the rotation-only RANSAC and the relative one:

    // create a RelativePoseSac problem and RANSAC
    typedef opengv::sac_problems::relative_pose::FrameRotationOnlySacProblem
        FrameRotationOnlySacProblem;
    opengv::sac::Ransac<FrameRotationOnlySacProblem> rotation_only_ransac;
    std::shared_ptr<FrameRotationOnlySacProblem> rotation_only_problem_ptr(
          new FrameRotationOnlySacProblem(adapter));
    rotation_only_ransac.sac_model_ = rotation_only_problem_ptr;
    rotation_only_ransac.threshold_ = 9;
    rotation_only_ransac.max_iterations_ = 50;

    // run the ransac
    rotation_only_ransac.computeModel(0);

    // get quality
    int rotation_only_inliers = int(rotation_only_ransac.inliers_.size());
    float rotation_only_ratio = float(rotation_only_inliers) / float(numCorrespondences);

    // now the rel_pose one:
    typedef opengv::sac_problems::relative_pose::FrameRelativePoseSacProblem
        FrameRelativePoseSacProblem;
    opengv::sac::Ransac<FrameRelativePoseSacProblem> rel_pose_ransac;
    std::shared_ptr<FrameRelativePoseSacProblem> rel_pose_problem_ptr(
          new FrameRelativePoseSacProblem(adapter, FrameRelativePoseSacProblem::STEWENIUS));
    rel_pose_ransac.sac_model_ = rel_pose_problem_ptr;
    rel_pose_ransac.threshold_ = 9;  //(1.0 - cos(0.5/600));
    rel_pose_ransac.max_iterations_ = 50;

    // run the ransac
    rel_pose_ransac.computeModel(0);

    // assess success
    int rel_pose_inliers = int(rel_pose_ransac.inliers_.size());
    float rel_pose_ratio = float(rel_pose_inliers) / float(numCorrespondences);

    // decide on success and fill inliers
    std::vector<bool> inliers(numCorrespondences, false);
    if (rotation_only_ratio > rel_pose_ratio || rotation_only_ratio > 0.8f) {
      if (rotation_only_inliers > 10) {
        rotation_only_success = true;
      }
      rotationOnly = true;
      totalInlierNumber += rotation_only_inliers;
      for (size_t k = 0; k < rotation_only_ransac.inliers_.size(); ++k) {
        inliers.at(size_t(rotation_only_ransac.inliers_.at(k))) = true;
      }
    } else {
      if (rel_pose_inliers > 10 && rel_pose_ratio > 0.8f) {
        rel_pose_success = true;
      }
      totalInlierNumber += rel_pose_inliers;
      for (size_t k = 0; k < rel_pose_ransac.inliers_.size(); ++k) {
        inliers.at(size_t(rel_pose_ransac.inliers_.at(k))) = true;
      }
    }

    // failure?
    if (!rotation_only_success && !rel_pose_success) {
      continue;
    }

    // otherwise: kick out outliers!
    std::shared_ptr<okvis::MultiFrame> multiFrame = estimator.multiFrame(StateId(currentFrameId));
    for (size_t k = 0; k < numCorrespondences; ++k) {
      size_t idxB = adapter.getMatchKeypointIdxB(k);
      if (removeOutliers && !inliers[k]) {
        uint64_t lmIdB = multiFrame->landmarkId(im, idxB);
        if(lmIdB !=0) {
          estimator.removeObservation(StateId(currentFrameId), im, idxB);
        }
      }
    }

    // initialize pose if necessary
    if (initializePose && !isInitialized_) {
      if (rel_pose_success) {
        //LOG(INFO) << "Initializing pose from 2D-2D RANSAC"; #Sebastian
      } else {
        //LOG(INFO) << "Initializing pose from 2D-2D RANSAC: orientation only";
      }
    }
  }

  if (rel_pose_success || rotation_only_success) {
    return totalInlierNumber;
  }

  rotationOnly = true;  // hack...
  return -1;

}

// (re)instantiates feature detectors and descriptor extractors. Used after settings changed or at
// startup.
void Frontend::initialiseBriskFeatureDetectors() {
  for (auto it = featureDetectorMutexes_.begin(); it != featureDetectorMutexes_.end(); ++it) {
    (*it)->lock();
  }
  //mClahe = cv::createCLAHE(3.0, cv::Size(8, 8));
  featureDetectors_.clear();
  descriptorExtractors_.clear();
  for (size_t i = 0; i < numCameras_; ++i) {
    featureDetectors_.push_back(std::shared_ptr<cv::FeatureDetector>(
        new brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator>(
            briskDetectionThreshold_, briskDetectionOctaves_,
            briskDetectionAbsoluteThreshold_, briskDetectionMaximumKeypoints_)));
    descriptorExtractors_.push_back(std::shared_ptr<cv::DescriptorExtractor>(
        new brisk::BriskDescriptorExtractor(
            briskDescriptionRotationInvariance_, briskDescriptionScaleInvariance_)));
  }
  for (auto it = featureDetectorMutexes_.begin(); it != featureDetectorMutexes_.end(); ++it) {
    (*it)->unlock();
  }
}

}  // namespace okvis
