/**
 * @file mowe_reloc_eval.cpp
 * @brief Mow-e T-0121: boot-relocalisation harness. Loads a .mowemap built on one
 *        TUM-VI sequence (room1), picks --attempts random start frames of another
 *        sequence in the same mocap room (room2) and runs mowe_map::Relocaliser on
 *        consecutive frames (--stride, --max-frames) until it is confident. Ground
 *        truth: both mocap files share the mocap frame M; the map's W is aligned to M
 *        with Eigen::umeyama on the map keyframes vs the map dataset's mocap, and the
 *        residual constant body offset T_RS (mocap body vs IMU) is estimated the same
 *        way, so expected T_WS(t) = T_WM * T_MR(t) * T_RS.
 *
 *   mowe_reloc_eval --map m.mowemap --dataset room2/mav0 --attempts 20 --json out.json
 *                   [--config okvis.yaml]        default: the map header's config_path
 *                   [--map-dataset room1/mav0]   default: the map header's dataset_path
 *                   [--seed 0] [--stride 5] [--max-frames 12] [--max-dt-ms 10]
 *                   [--agree-pos-m 1.0] [--agree-rot-deg 60] [--v 1 (per-frame trace)]
 *
 *   JSON: success_rate (confident and <= 1 m), median_position_err_m, median_yaw_err_deg,
 *   median_rot_err_deg, false_confident (confident but > 1 m), map_load_ms, map_size_mb,
 *   alignment residuals, per-attempt records.
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <glog/logging.h>
#include <opencv2/imgcodecs.hpp>
#include <sys/stat.h>

#include <mowe_map/Relocaliser.hpp>
#include <mowe_map/mowemap.hpp>
#include <mowe_vpr/embedder.hpp>
#include <mowe_vpr/vlad.hpp>
#include <okvis/Frontend.hpp>
#include <okvis/ViParametersReader.hpp>
#include <okvis/xfeat/LighterGlueMatcher.hpp>

namespace {
using okvis::kinematics::Transformation;
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

struct Pose { int64_t t_ns; Transformation T; };

// TUM-VI mocap0/data.csv: timestamp[ns], p x y z, q w x y z
std::vector<Pose> readMocap(const std::string& path) {
  std::vector<Pose> out;
  std::ifstream f(path);
  CHECK(f.good()) << "cannot read " << path;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream ss(line);
    long long t; double p[3], q[4];
    if (!(ss >> t >> p[0] >> p[1] >> p[2] >> q[0] >> q[1] >> q[2] >> q[3])) continue;
    out.push_back({int64_t(t), Transformation(Eigen::Vector3d(p[0], p[1], p[2]),
                                             Eigen::Quaterniond(q[0], q[1], q[2], q[3]).normalized())});
  }
  std::sort(out.begin(), out.end(), [](const Pose& a, const Pose& b) { return a.t_ns < b.t_ns; });
  return out;
}
// nearest mocap pose within maxDtNs; false if none.
bool lookup(const std::vector<Pose>& gt, int64_t t_ns, int64_t maxDtNs, Transformation& T) {
  auto it = std::lower_bound(gt.begin(), gt.end(), t_ns, [](const Pose& p, int64_t t) { return p.t_ns < t; });
  const Pose* best = nullptr;
  if (it != gt.end()) best = &*it;
  if (it != gt.begin() && (!best || std::llabs((it - 1)->t_ns - t_ns) < std::llabs(best->t_ns - t_ns))) best = &*(it - 1);
  if (!best || std::llabs(best->t_ns - t_ns) > maxDtNs) return false;
  T = best->T;
  return true;
}
// cam0/data.csv: timestamp[ns], filename
std::vector<std::pair<int64_t, std::string>> readCam(const std::string& mav0, int cam) {
  std::vector<std::pair<int64_t, std::string>> out;
  const std::string dir = mav0 + "/cam" + std::to_string(cam);
  std::ifstream f(dir + "/data.csv");
  CHECK(f.good()) << "cannot read " << dir << "/data.csv";
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    const size_t c = line.find(',');
    if (c == std::string::npos) continue;
    std::string name = line.substr(c + 1);
    while (!name.empty() && (name.back() == '\r' || name.back() == ' ')) name.pop_back();
    out.emplace_back(std::stoll(line.substr(0, c)), dir + "/data/" + name);
  }
  return out;
}
double angleDeg(const Eigen::Quaterniond& q) { return 2.0 * std::atan2(q.vec().norm(), std::abs(q.w())) * 180.0 / M_PI; }
double median(std::vector<double> v) {
  if (v.empty()) return std::nan("");
  std::sort(v.begin(), v.end());
  return v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}
std::string arg(int argc, char** argv, const std::string& key, const std::string& def = "") {
  for (int i = 1; i + 1 < argc; ++i) if (argv[i] == key) return argv[i + 1];
  return def;
}
std::string jnum(double v) {  // JSON has no NaN
  if (!std::isfinite(v)) return "null";
  char b[64]; snprintf(b, sizeof b, "%.6g", v); return b;
}
}  // namespace

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  FLAGS_v = std::stoi(arg(argc, argv, "--v", "0"));
  const std::string mapPath = arg(argc, argv, "--map"), dataset = arg(argc, argv, "--dataset");
  const std::string jsonPath = arg(argc, argv, "--json", "reloc.json");
  const int attempts = std::stoi(arg(argc, argv, "--attempts", "20"));
  const int stride = std::stoi(arg(argc, argv, "--stride", "5"));
  const int maxFrames = std::stoi(arg(argc, argv, "--max-frames", "12"));
  const unsigned seed = unsigned(std::stoul(arg(argc, argv, "--seed", "0")));
  const int64_t maxDtNs = int64_t(std::stod(arg(argc, argv, "--max-dt-ms", "10")) * 1e6);
  if (mapPath.empty() || dataset.empty()) {
    std::cerr << "usage: mowe_reloc_eval --map m.mowemap --dataset mav0 [--attempts N] [--json out.json] "
                 "[--config yaml] [--map-dataset mav0] [--seed S] [--stride K] [--max-frames M]\n";
    return 2;
  }

  // ---- map: load (timed) ---------------------------------------------------------------
  const auto tLoad0 = Clock::now();
  mowe_map::Map map;
  std::string err;
  if (!map.load(mapPath, &err)) { LOG(ERROR) << err; return 1; }
  const double loadFileMs = ms(tLoad0, Clock::now());
  struct stat st{};
  CHECK_EQ(stat(mapPath.c_str(), &st), 0);
  const double mapSizeMb = double(st.st_size) / 1e6;

  const std::string configPath = arg(argc, argv, "--config", map.header.config_path);
  const std::string mapDataset = arg(argc, argv, "--map-dataset", map.header.dataset_path);
  okvis::ViParametersReader reader(configPath);
  okvis::ViParameters params;
  reader.getParameters(params);
  const okvis::cameras::NCameraSystem& cams = params.nCameraSystem;
  CHECK_EQ(cams.numCameras(), 2u) << "stereo rig expected";
  CHECK(params.frontend.xfeat.use && !params.frontend.vpr.engine.empty())
      << configPath << " needs frontend_parameters.xfeat + vpr";

  // ---- identity checks (docs/design/mowemap-format.md) --------------------------------
  const uint64_t xfeatHash = mowe_map::hash_file(params.frontend.xfeat.engine);
  const uint64_t vprHash = mowe_map::hash_file(params.frontend.vpr.engine);
  const uint64_t vocHash = mowe_map::hash_file(params.frontend.vpr.vocabulary);
  const uint64_t calHash = okvis::Frontend::calibrationHash(cams);
  if (xfeatHash != map.header.xfeat_engine_hash) { LOG(ERROR) << "map built with a different XFeat engine"; return 1; }
  if (vprHash != map.header.vpr_engine_hash) { LOG(ERROR) << "map built with a different VPR engine"; return 1; }
  if (vocHash != map.header.vocabulary_hash) { LOG(ERROR) << "map built with a different VPR vocabulary"; return 1; }
  if (calHash != map.header.calibration_hash) { LOG(ERROR) << "map built with a different camera calibration"; return 1; }

  // ---- engines --------------------------------------------------------------------------
  okvis::Frontend frontend(cams.numCameras(), "");
  frontend.setBriskDetectionMaximumKeypoints(size_t(params.frontend.max_num_keypoints));
  frontend.setXFeatParameters(params.frontend.xfeat);
  mowe_vpr::TrtEmbedder embedder;
  CHECK(embedder.load(params.frontend.vpr.engine)) << params.frontend.vpr.engine;
  mowe_vpr::Vocabulary vocabulary;
  CHECK(vocabulary.load(params.frontend.vpr.vocabulary)) << params.frontend.vpr.vocabulary;
  CHECK_EQ(vocabulary.desc_dim(), int(map.header.vpr_dim));
  okvis::xfeat::LighterGlueConfig lgCfg;
  lgCfg.engine_path = params.frontend.xfeat.lighterglue_engine;
  lgCfg.min_score = float(params.frontend.xfeat.match_score_min);
  okvis::xfeat::LighterGlueMatcher lighterGlue(lgCfg);
  CHECK(lighterGlue.loaded()) << "LighterGlue engine " << lgCfg.engine_path;

  mowe_map::RelocaliserParams rp;
  rp.top_k = params.frontend.vpr.top_k;
  rp.score_min = params.frontend.vpr.score_min;
  rp.min_inliers = params.frontend.vpr.min_inliers;
  rp.reproj_px = params.frontend.vpr.reproj_px;
  rp.min_inlier_ratio = params.frontend.vpr.min_inlier_ratio;
  rp.consecutive_required = params.frontend.vpr.consecutive_required;
  rp.agree_pos_m = std::stod(arg(argc, argv, "--agree-pos-m", "1.0"));
  rp.agree_rot_deg = std::stod(arg(argc, argv, "--agree-rot-deg", "60"));
  rp.keypoint_size_px = params.frontend.xfeat.keypoint_size;
  const auto tDb0 = Clock::now();
  mowe_map::Relocaliser reloc(map, cams, lighterGlue, rp);
  const double loadDbMs = ms(tDb0, Clock::now());
  LOG(INFO) << "map " << mapPath << ": " << map.keyframes.size() << " keyframes, " << map.landmarks.size()
            << " landmarks, " << map.edges.size() << " edges, " << mapSizeMb << " MB; file " << loadFileMs
            << " ms + database " << loadDbMs << " ms";

  // ---- ground truth: align the map's W to the mocap frame M --------------------------------
  const auto gtMap = readMocap(mapDataset + "/mocap0/data.csv");
  const auto gtNew = readMocap(dataset + "/mocap0/data.csv");
  std::vector<Eigen::Vector3d> src, dst;
  std::vector<std::pair<Transformation, Transformation>> pairs;  // (T_MR, T_WS)
  for (size_t i = 0; i < map.keyframes.size(); ++i) {
    Transformation T_MR;
    if (!lookup(gtMap, map.keyframes[i].t_ns, maxDtNs, T_MR)) continue;
    const Transformation& T_WS = reloc.keyframePose(i);
    src.push_back(T_WS.r());
    dst.push_back(T_MR.r());
    pairs.emplace_back(T_MR, T_WS);
  }
  CHECK_GE(src.size(), 10u) << "too few map keyframes with mocap";
  Eigen::Matrix3Xd S(3, src.size()), D(3, dst.size());
  for (size_t i = 0; i < src.size(); ++i) { S.col(i) = src[i]; D.col(i) = dst[i]; }
  const Transformation T_MW(Eigen::Matrix4d(Eigen::umeyama(S, D, false)));
  // constant body offset mocap body R -> IMU S (identity if the mocap is already in the IMU frame):
  // chordal mean of T_MR^-1 * T_MW * T_WS over the keyframes.
  Eigen::Vector3d rSum = Eigen::Vector3d::Zero();
  Eigen::Matrix4d qSum = Eigen::Matrix4d::Zero();
  for (const auto& pr : pairs) {
    const Transformation T_RS = pr.first.inverse() * T_MW * pr.second;
    rSum += T_RS.r();
    const Eigen::Vector4d qc = T_RS.q().coeffs();
    qSum += qc * qc.transpose();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> es(qSum);
  const Eigen::Vector4d qMean = es.eigenvectors().col(3);
  const Transformation T_RS(rSum / double(pairs.size()), Eigen::Quaterniond(qMean[3], qMean[0], qMean[1], qMean[2]).normalized());
  const Transformation T_WM = T_MW.inverse();
  std::vector<double> alignPos, alignRot;
  for (const auto& pr : pairs) {
    const Transformation d = (T_WM * pr.first * T_RS).inverse() * pr.second;
    alignPos.push_back(d.r().norm());
    alignRot.push_back(angleDeg(d.q()));
  }
  LOG(INFO) << "map W -> mocap alignment on " << pairs.size() << " keyframes: residual pos rmse "
            << std::sqrt(std::inner_product(alignPos.begin(), alignPos.end(), alignPos.begin(), 0.0) / alignPos.size())
            << " m, median rot " << median(alignRot) << " deg; body offset |t_RS| " << T_RS.r().norm() << " m, "
            << angleDeg(T_RS.q()) << " deg";

  // ---- dataset frames -----------------------------------------------------------------------
  const auto cam0 = readCam(dataset, 0), cam1 = readCam(dataset, 1);
  std::vector<std::pair<int64_t, std::pair<std::string, std::string>>> frames;
  for (size_t i = 0, j = 0; i < cam0.size() && j < cam1.size();) {
    if (cam0[i].first == cam1[j].first) { frames.push_back({cam0[i].first, {cam0[i].second, cam1[j].second}}); ++i; ++j; }
    else if (cam0[i].first < cam1[j].first) ++i; else ++j;
  }
  CHECK_GT(int(frames.size()), stride * maxFrames + 1) << "dataset too short";

  // ---- attempts -------------------------------------------------------------------------------
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> pick(0, int(frames.size()) - stride * maxFrames - 1);
  struct Rec { int start; bool confident; int framesUsed; int kf; float score; int inliers; int corr; bool gt; double posErr, yawErr, rotErr; double perFrameMs;
               int framesVerified = 0; int framesGated = 0; float maxTopScore = 0.f; int maxInliers = 0; };
  std::vector<Rec> recs;
  std::vector<double> extractMs, embedMs, pushMs;
  uint64_t frameId = 1;
  for (int a = 0; a < attempts; ++a) {
    const int start = pick(rng);
    reloc.reset();
    Rec rec{start, false, 0, -1, 0.f, 0, 0, false, std::nan(""), std::nan(""), std::nan(""), 0.0};
    const auto tA0 = Clock::now();
    for (int j = 0; j < maxFrames; ++j) {
      const auto& fr = frames[size_t(start + j * stride)];
      cv::Mat left = cv::imread(fr.second.first, cv::IMREAD_GRAYSCALE);   // 16-bit PNG -> 8-bit, as DatasetReader
      cv::Mat right = cv::imread(fr.second.second, cv::IMREAD_GRAYSCALE);
      CHECK(!left.empty() && !right.empty()) << fr.second.first;
      auto mf = std::make_shared<okvis::MultiFrame>(cams, okvis::Time().fromNSec(uint64_t(fr.first)), frameId++);
      mf->setImage(0, left);
      mf->setImage(1, right);
      const auto t0 = Clock::now();
      frontend.detectAndDescribe(0, mf, Transformation(), nullptr);  // XFeat pair path does both cameras
      const auto t1 = Clock::now();
      Eigen::VectorXf cls;
      Eigen::MatrixXf patch;
      CHECK(embedder.embed(left, cls, patch));
      patch.rowwise().normalize();  // the vocabulary was fitted on normalised tokens (T-0119/T-0120)
      const Eigen::VectorXf desc = mowe_vpr::describe(patch, vocabulary);
      const auto t2 = Clock::now();
      const mowe_map::Relocalisation r = reloc.push(mf, desc);
      const auto t3 = Clock::now();
      extractMs.push_back(ms(t0, t1)); embedMs.push_back(ms(t1, t2)); pushMs.push_back(ms(t2, t3));
      rec.framesUsed = j + 1;
      rec.framesVerified += r.verified ? 1 : 0;
      rec.framesGated += r.candidates > 0 ? 1 : 0;
      rec.maxTopScore = std::max(rec.maxTopScore, r.top_score);
      rec.maxInliers = std::max(rec.maxInliers, r.inliers);
      VLOG(1) << "  attempt " << a << " frame " << j << " (idx " << start + j * stride << "): top kf " << r.top_keyframe << " score " << r.top_score
              << ", " << r.candidates << " gated, " << (r.verified ? "VERIFIED kf " + std::to_string(r.keyframe_index) + " inliers " + std::to_string(r.inliers) + "/" + std::to_string(r.correspondences) : "not verified")
              << (r.confident ? " -> CONFIDENT" : "");
      if (r.confident) {
        rec.confident = true; rec.kf = r.keyframe_index; rec.score = r.vpr_score; rec.inliers = r.inliers; rec.corr = r.correspondences;
        Transformation T_MR;
        if (lookup(gtNew, fr.first, maxDtNs, T_MR)) {
          rec.gt = true;
          const Transformation T_WS_gt = T_WM * T_MR * T_RS;
          rec.posErr = (r.T_WS.r() - T_WS_gt.r()).norm();
          const Eigen::Matrix3d Rerr = r.T_WS.C() * T_WS_gt.C().transpose();  // world-frame error rotation
          rec.yawErr = std::abs(std::atan2(Rerr(1, 0), Rerr(0, 0))) * 180.0 / M_PI;
          rec.rotErr = angleDeg(Eigen::Quaterniond(Rerr));
        }
        break;
      }
    }
    rec.perFrameMs = ms(tA0, Clock::now()) / std::max(1, rec.framesUsed);
    LOG(INFO) << "attempt " << a << " start frame " << start << ": " << (rec.confident ? "CONFIDENT" : "not confident")
              << " after " << rec.framesUsed << " frames (" << rec.framesGated << " gated, " << rec.framesVerified << " verified, max top score "
              << rec.maxTopScore << ", max inliers " << rec.maxInliers << ")" << (rec.confident ? " kf " + std::to_string(rec.kf) + " score " + std::to_string(rec.score) +
              " inliers " + std::to_string(rec.inliers) + "/" + std::to_string(rec.corr) + (rec.gt ? " err " + std::to_string(rec.posErr) + " m / " +
              std::to_string(rec.yawErr) + " deg yaw" : " (no mocap)") : "");
    recs.push_back(rec);
  }

  // ---- report ---------------------------------------------------------------------------------
  int successes = 0, falseConfident = 0, confidentNoGt = 0;
  std::vector<double> pos, yaw, rot, used;
  for (const Rec& r : recs) {
    if (!r.confident) continue;
    if (!r.gt) { ++confidentNoGt; continue; }
    if (r.posErr <= 1.0) { ++successes; pos.push_back(r.posErr); yaw.push_back(r.yawErr); rot.push_back(r.rotErr); used.push_back(r.framesUsed); }
    else ++falseConfident;
  }
  auto mean = [](const std::vector<double>& v) { return v.empty() ? std::nan("") : std::accumulate(v.begin(), v.end(), 0.0) / v.size(); };
  std::ofstream js(jsonPath);
  js << "{\n  \"map\": \"" << mapPath << "\",\n  \"dataset\": \"" << dataset << "\",\n  \"map_dataset\": \"" << mapDataset << "\",\n"
     << "  \"config\": \"" << configPath << "\",\n  \"attempts\": " << attempts << ",\n  \"seed\": " << seed << ",\n  \"stride\": " << stride
     << ",\n  \"max_frames\": " << maxFrames << ",\n"
     << "  \"success_rate\": " << jnum(double(successes) / std::max(1, attempts)) << ",\n  \"successes\": " << successes
     << ",\n  \"false_confident\": " << falseConfident << ",\n  \"confident_without_gt\": " << confidentNoGt << ",\n"
     << "  \"median_position_err_m\": " << jnum(median(pos)) << ",\n  \"median_yaw_err_deg\": " << jnum(median(yaw))
     << ",\n  \"median_rot_err_deg\": " << jnum(median(rot)) << ",\n  \"max_position_err_m\": " << jnum(pos.empty() ? std::nan("") : *std::max_element(pos.begin(), pos.end()))
     << ",\n  \"mean_frames_to_confident\": " << jnum(mean(used)) << ",\n"
     << "  \"map_load_ms\": " << jnum(loadFileMs + loadDbMs) << ",\n  \"map_load_file_ms\": " << jnum(loadFileMs) << ",\n  \"map_load_database_ms\": " << jnum(loadDbMs)
     << ",\n  \"map_size_mb\": " << jnum(mapSizeMb) << ",\n  \"map_keyframes\": " << map.keyframes.size() << ",\n  \"map_landmarks\": " << map.landmarks.size()
     << ",\n  \"map_edges\": " << map.edges.size() << ",\n"
     << "  \"alignment\": {\"keyframes\": " << pairs.size() << ", \"pos_rmse_m\": "
     << jnum(std::sqrt(std::inner_product(alignPos.begin(), alignPos.end(), alignPos.begin(), 0.0) / alignPos.size()))
     << ", \"median_rot_deg\": " << jnum(median(alignRot)) << ", \"body_offset_m\": " << jnum(T_RS.r().norm()) << ", \"body_offset_deg\": " << jnum(angleDeg(T_RS.q())) << "},\n"
     << "  \"timing_ms\": {\"xfeat_pair_mean\": " << jnum(mean(extractMs)) << ", \"vpr_embed_mean\": " << jnum(mean(embedMs)) << ", \"relocaliser_push_mean\": " << jnum(mean(pushMs))
     << ", \"relocaliser_push_max\": " << jnum(pushMs.empty() ? std::nan("") : *std::max_element(pushMs.begin(), pushMs.end())) << "},\n"
     << "  \"params\": {\"top_k\": " << rp.top_k << ", \"score_min\": " << rp.score_min << ", \"min_inliers\": " << rp.min_inliers << ", \"reproj_px\": " << rp.reproj_px
     << ", \"min_inlier_ratio\": " << rp.min_inlier_ratio << ", \"consecutive_required\": " << rp.consecutive_required
     << ", \"agree_pos_m\": " << rp.agree_pos_m << ", \"agree_rot_deg\": " << rp.agree_rot_deg << "},\n"
     << "  \"records\": [";
  for (size_t i = 0; i < recs.size(); ++i) {
    const Rec& r = recs[i];
    js << (i ? ",\n    " : "\n    ") << "{\"start\": " << r.start << ", \"t_ns\": " << frames[size_t(r.start)].first << ", \"confident\": " << (r.confident ? "true" : "false")
       << ", \"frames\": " << r.framesUsed << ", \"keyframe\": " << r.kf << ", \"score\": " << jnum(r.score) << ", \"inliers\": " << r.inliers << ", \"correspondences\": " << r.corr
       << ", \"frames_gated\": " << r.framesGated << ", \"frames_verified\": " << r.framesVerified << ", \"max_top_score\": " << jnum(r.maxTopScore) << ", \"max_inliers\": " << r.maxInliers
       << ", \"gt\": " << (r.gt ? "true" : "false") << ", \"pos_err_m\": " << jnum(r.posErr) << ", \"yaw_err_deg\": " << jnum(r.yawErr) << ", \"rot_err_deg\": " << jnum(r.rotErr)
       << ", \"ms_per_frame\": " << jnum(r.perFrameMs) << "}";
  }
  js << "\n  ]\n}\n";
  LOG(INFO) << "success " << successes << "/" << attempts << ", false_confident " << falseConfident << ", median pos "
            << median(pos) << " m, median yaw " << median(yaw) << " deg, load " << loadFileMs + loadDbMs << " ms, size " << mapSizeMb
            << " MB -> " << jsonPath;
  return 0;
}
