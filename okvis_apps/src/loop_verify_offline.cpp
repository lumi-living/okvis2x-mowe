/**
 * @file loop_verify_offline.cpp
 * @brief Mow-e T-0120: offline sweep of the loop-closure geometric verification
 *        on a saved keyframes.bin. Picks revisit pairs (poses within --max-dist
 * m / --max-deg, at least --min-gap keyframes apart), matches them with
 *        LighterGlue exactly like Frontend::verifyRecognisedPlace and runs the
 *        same GP3P RANSAC (okvis::loopclosure::ransacAbsolutePose) at several
 *        reprojection thresholds. Prints, per threshold, how many pairs would
 *        pass min_inliers + the 0.7 inlier-ratio rule — the knob-setting
 *        evidence for frontend_parameters.vpr.reproj_px.
 *
 *   loop_verify_offline config.yaml keyframes.bin [--max-dist 1.0] [--max-deg
 * 30]
 *                       [--min-gap 30] [--max-pairs 200] [--min-inliers 20] [--min-ratio 0.7]
 *                       [--thresholds 2,3,4,6,8]
 */

#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <okvis/DescriptorDistance.hpp>
#include <okvis/LoopClosureGates.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/ViParametersReader.hpp>
#include <okvis/xfeat/LighterGlueMatcher.hpp>

namespace {
template <typename T> bool rd(FILE *f, T &v) {
  return std::fread(&v, sizeof(T), 1, f) == 1;
}

struct Kf {
  uint64_t id;
  okvis::kinematics::Transformation T_WS;
  std::shared_ptr<okvis::MultiFrame> mf;
  size_t numLandmarks = 0;
};

std::vector<Kf> load(const std::string &path,
                     const okvis::cameras::NCameraSystem &cams) {
  std::vector<Kf> out;
  FILE *f = std::fopen(path.c_str(), "rb");
  CHECK(f) << "cannot open " << path;
  char magic[8];
  uint32_t numCameras, vprDim, numKeyframes;
  CHECK(std::fread(magic, 1, 8, f) == 8 &&
        std::memcmp(magic, "MOWEKF01", 8) == 0);
  CHECK(rd(f, numCameras) && rd(f, vprDim) && rd(f, numKeyframes));
  CHECK_EQ(numCameras, cams.numCameras());
  for (uint32_t n = 0; n < numKeyframes; ++n) {
    Kf kf;
    int64_t t_ns;
    double r[3], q[4];
    CHECK(rd(f, kf.id) && rd(f, t_ns));
    for (double &v : r)
      CHECK(rd(f, v));
    for (double &v : q)
      CHECK(rd(f, v));
    kf.T_WS = okvis::kinematics::Transformation(
        Eigen::Vector3d(r[0], r[1], r[2]),
        Eigen::Quaterniond(q[3], q[0], q[1], q[2]));
    std::vector<float> desc(vprDim);
    CHECK(std::fread(desc.data(), 4, vprDim, f) == vprDim);
    kf.mf.reset(new okvis::MultiFrame(
        cams, okvis::Time().fromNSec(uint64_t(t_ns)), kf.id));
    for (uint32_t im = 0; im < numCameras; ++im) {
      uint32_t K;
      CHECK(rd(f, K));
      std::vector<cv::KeyPoint> kps(K);
      cv::Mat d(int(K), okvis::kFloatDescriptorDim, CV_32FC1);
      std::vector<uint64_t> lmIds(K);
      std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
          hps(K);
      std::vector<bool> init(K);
      for (uint32_t k = 0; k < K; ++k) {
        float x, y, size, response;
        CHECK(rd(f, x) && rd(f, y) && rd(f, size) && rd(f, response));
        kps[k] = cv::KeyPoint(x, y, size, -1.f, response);
        CHECK(std::fread(d.ptr<float>(int(k)), 4, okvis::kFloatDescriptorDim,
                         f) == size_t(okvis::kFloatDescriptorDim));
        uint8_t i8;
        CHECK(rd(f, lmIds[k]) && rd(f, i8));
        for (int i = 0; i < 4; ++i)
          CHECK(rd(f, hps[k][i]));
        init[k] = i8 != 0;
      }
      kf.mf->resetKeypoints(im, kps);
      kf.mf->resetDescriptors(im, d);
      kf.mf->computeBackProjections(im);
      for (uint32_t k = 0; k < K; ++k) {
        kf.mf->setLandmarkId(im, k, lmIds[k]);
        kf.mf->setLandmark(im, k, hps[k], init[k]);
        if (init[k])
          ++kf.numLandmarks;
      }
    }
    out.push_back(kf);
  }
  std::fclose(f);
  return out;
}

// old -> new LighterGlue proposals per camera (mirrors
// Frontend::lighterGluePairProposals)
std::vector<int> proposals(okvis::xfeat::LighterGlueMatcher &lg,
                           const okvis::MultiFrame &a, size_t im,
                           const okvis::MultiFrame &b) {
  const size_t nA = a.numKeypoints(im), nB = b.numKeypoints(im);
  std::vector<int> out(nA, -1);
  if (!nA || !nB)
    return out;
  std::vector<float> ka(nA * 2), kb(nB * 2), sa(nA), sb(nB);
  cv::KeyPoint kp;
  for (size_t k = 0; k < nA; ++k) {
    a.getCvKeypoint(im, k, kp);
    ka[2 * k] = kp.pt.x;
    ka[2 * k + 1] = kp.pt.y;
    sa[k] = kp.response;
  }
  for (size_t k = 0; k < nB; ++k) {
    b.getCvKeypoint(im, k, kp);
    kb[2 * k] = kp.pt.x;
    kb[2 * k + 1] = kp.pt.y;
    sb[k] = kp.response;
  }
  const auto m = lg.match(
      ka.data(), reinterpret_cast<const float *>(a.keypointDescriptor(im, 0)),
      sa.data(), nA, uint32_t(a.geometry(im)->imageWidth()),
      uint32_t(a.geometry(im)->imageHeight()), kb.data(),
      reinterpret_cast<const float *>(b.keypointDescriptor(im, 0)), sb.data(),
      nB, uint32_t(b.geometry(im)->imageWidth()),
      uint32_t(b.geometry(im)->imageHeight()));
  for (size_t i = 0; i < m.size(); ++i)
    out[m.indices[i].first] = int(m.indices[i].second);
  return out;
}
} // namespace

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  if (argc < 3) {
    std::cerr
        << "usage: loop_verify_offline config.yaml keyframes.bin [options]\n";
    return 1;
  }
  double maxDist = 1.0, maxDeg = 30.0;
  int minGap = 30, maxPairs = 200, minInliers = 20;
  double minRatio = 0.7;
  std::vector<double> thresholds = {2, 3, 4, 6, 8};
  for (int i = 3; i + 1 < argc; i += 2) {
    const std::string a = argv[i], v = argv[i + 1];
    if (a == "--max-dist")
      maxDist = std::stod(v);
    else if (a == "--max-deg")
      maxDeg = std::stod(v);
    else if (a == "--min-gap")
      minGap = std::stoi(v);
    else if (a == "--max-pairs")
      maxPairs = std::stoi(v);
    else if (a == "--min-inliers")
      minInliers = std::stoi(v);
    else if (a == "--min-ratio")
      minRatio = std::stod(v);
    else if (a == "--thresholds") {
      thresholds.clear();
      size_t p = 0;
      while (p < v.size()) {
        size_t q = v.find(',', p);
        thresholds.push_back(std::stod(v.substr(p, q - p)));
        if (q == std::string::npos)
          break;
        p = q + 1;
      }
    }
  }
  okvis::ViParametersReader reader(argv[1]);
  okvis::ViParameters params;
  reader.getParameters(params);
  const auto kfs = load(argv[2], params.nCameraSystem);
  LOG(INFO) << kfs.size() << " keyframes";
  okvis::xfeat::LighterGlueConfig cfg;
  cfg.engine_path = params.frontend.xfeat.lighterglue_engine;
  cfg.min_score = float(params.frontend.xfeat.match_score_min);
  okvis::xfeat::LighterGlueMatcher lg(cfg);
  CHECK(lg.loaded()) << "LighterGlue engine";
  const double keypointSize = params.frontend.xfeat.keypoint_size;

  // revisit pairs
  std::vector<std::pair<size_t, size_t>> pairs;
  for (size_t j = 0; j < kfs.size() && int(pairs.size()) < maxPairs; ++j) {
    for (size_t i = 0; i + size_t(minGap) < j; ++i) {
      const okvis::kinematics::Transformation dT =
          kfs[i].T_WS.inverse() * kfs[j].T_WS;
      const double ang = 2.0 *
                         std::atan2(dT.q().vec().norm(), std::abs(dT.q().w())) *
                         180.0 / M_PI;
      if (dT.r().norm() <= maxDist && ang <= maxDeg) {
        pairs.emplace_back(i, j);
        break;
      } // one old per new
    }
  }
  LOG(INFO) << pairs.size() << " revisit pairs (<= " << maxDist
            << " m, <= " << maxDeg << " deg, gap >= " << minGap << ")";

  struct Acc {
    int pass = 0;
    double inl = 0, ratio = 0;
  };
  std::map<double, Acc> acc;
  double corrSum = 0, lgSum = 0;
  for (const auto &pr : pairs) {
    const Kf &oldKf = kfs[pr.first];
    const Kf &newKf = kfs[pr.second];
    okvis::AlignedMap<uint64_t, Eigen::Vector4d> points;
    std::map<okvis::KeypointIdentifier, uint64_t> matches;
    size_t lgMatches = 0;
    for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im) {
      const auto prop = proposals(lg, *oldKf.mf, im, *newKf.mf);
      for (size_t kOld = 0; kOld < prop.size(); ++kOld) {
        if (prop[kOld] < 0)
          continue;
        ++lgMatches;
        const uint64_t lmId = oldKf.mf->landmarkId(im, kOld);
        Eigen::Vector4d hp;
        bool init = false;
        oldKf.mf->getLandmark(im, kOld, hp, init);
        if (!lmId || !init || hp.norm() < 1e-12)
          continue;
        const okvis::KeypointIdentifier kid(newKf.id, im, size_t(prop[kOld]));
        if (matches.count(kid))
          continue;
        points[lmId] = hp;
        matches[kid] = lmId;
      }
    }
    corrSum += double(matches.size());
    lgSum += double(lgMatches);
    for (double t : thresholds) {
      okvis::kinematics::Transformation T;
      std::vector<bool> mask;
      std::vector<size_t> c, k;
      const int n = okvis::loopclosure::ransacAbsolutePose(
          points, matches, newKf.mf,
          okvis::loopclosure::ransacThresholdFromPixels(t, keypointSize), 50, T,
          mask, c, k);
      const double ratio = mask.empty() ? 0.0 : double(n) / double(mask.size());
      Acc &a = acc[t];
      a.inl += n;
      a.ratio += ratio;
      if (n >= minInliers && ratio >= minRatio)
        ++a.pass;
    }
  }
  const double np = std::max<double>(1.0, double(pairs.size()));
  std::printf("pairs %zu  mean LighterGlue matches %.1f  mean landmark "
              "correspondences %.1f\n",
              pairs.size(), lgSum / np, corrSum / np);
  std::printf("reproj_px  mean_inliers  mean_ratio  pass(min_inliers>=%d & "
              "ratio>=%.2f)\n",
              minInliers, minRatio);
  for (const auto &kv : acc)
    std::printf("%8.1f  %12.1f  %10.3f  %d/%zu\n", kv.first, kv.second.inl / np,
                kv.second.ratio / np, kv.second.pass, pairs.size());
  return 0;
}
