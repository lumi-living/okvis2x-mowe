/**
 * @file lighterglue_demo.cpp
 * @brief XFeat → LighterGlue on a rectified stereo PNG pair, no ROS (T-0114):
 *
 *   lighterglue_demo --engine lg.plan --xfeat-engine xfeat.plan
 *                    --left L.png --right R.png [--iters 200] [--warmup 10]
 *                    [--nn-threshold 0.25] [--json out.json]
 *
 * Reports the LighterGlue match count, the epipolar inlier ratio (rectified
 * pair → |Δv| ≤ 2 px), the mutual-NN cosine match count on the same top-K
 * sets for comparison, mean/p99 wall time of one match() call, and whether
 * the engine's padding mask is applied — proven by an A/B run: the same sets
 * once padded with -1 slots and once with garbage rows carrying score -1
 * must yield identical matches (masked keys contribute nothing).
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "okvis/xfeat/FrameFeaturesUtil.hpp"
#include "okvis/xfeat/LighterGlueMatcher.hpp"
#include "okvis/xfeat/LighterGlueUtil.hpp"
#include "okvis/xfeat/XFeatFrontend.hpp"

namespace {

using okvis::xfeat::kDescriptorDim;

/// A contiguous feature set (subset of a StreamFeatures, top-K by score).
struct Set {
  std::vector<float> kp, sc, de;
  std::size_t n() const { return sc.size(); }
  void append(const Set& o) {
    kp.insert(kp.end(), o.kp.begin(), o.kp.end());
    sc.insert(sc.end(), o.sc.begin(), o.sc.end());
    de.insert(de.end(), o.de.begin(), o.de.end());
  }
};

Set take(const okvis::xfeat::StreamFeatures& s, std::size_t k) {
  Set out;
  for (std::uint32_t i : okvis::xfeat::top_k_by_score(s.scores.data(), s.size(), k)) {
    out.kp.push_back(s.keypoints_px[i].u);
    out.kp.push_back(s.keypoints_px[i].v);
    out.sc.push_back(s.scores[i]);
    out.de.insert(out.de.end(), &s.descriptors[i * kDescriptorDim],
                  &s.descriptors[i * kDescriptorDim] + kDescriptorDim);
  }
  return out;
}

/// Garbage rows: random in-image coords, random unit descriptors, score -1.
Set garbage(std::size_t n, std::uint32_t w, std::uint32_t h, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd;
  Set g;
  for (std::size_t i = 0; i < n; ++i) {
    g.kp.push_back(float(rng() % w));
    g.kp.push_back(float(rng() % h));
    g.sc.push_back(okvis::xfeat::kLighterGluePadScore);
    double acc = 0;
    std::vector<float> d(kDescriptorDim);
    for (float& v : d) { v = nd(rng); acc += double(v) * v; }
    for (float& v : d) v /= float(std::sqrt(acc));
    g.de.insert(g.de.end(), d.begin(), d.end());
  }
  return g;
}

using Pairs = std::vector<std::pair<std::uint32_t, std::uint32_t>>;

/// Brute-force mutual-NN on unit descriptors: accept when the cosine distance
/// 1 - a·b is below `threshold` (the frontend's matching_threshold on the
/// float path, T-0112/T-0113) and the pair is mutually nearest.
Pairs mutual_nn(const Set& a, const Set& b, float threshold) {
  const std::size_t na = a.n(), nb = b.n();
  std::vector<int> bestB(na, -1), bestA(nb, -1);
  std::vector<float> bestBd(na, 2.f), bestAd(nb, 2.f);
  for (std::size_t i = 0; i < na; ++i) {
    for (std::size_t j = 0; j < nb; ++j) {
      float dot = 0.f;
      for (std::uint32_t k = 0; k < kDescriptorDim; ++k)
        dot += a.de[i * kDescriptorDim + k] * b.de[j * kDescriptorDim + k];
      const float dist = 1.f - dot;
      if (dist < bestBd[i]) { bestBd[i] = dist; bestB[i] = int(j); }
      if (dist < bestAd[j]) { bestAd[j] = dist; bestA[j] = int(i); }
    }
  }
  Pairs out;
  for (std::size_t i = 0; i < na; ++i)
    if (bestB[i] >= 0 && bestBd[i] < threshold && bestA[std::size_t(bestB[i])] == int(i))
      out.emplace_back(std::uint32_t(i), std::uint32_t(bestB[i]));
  return out;
}

/// Rectified pair: a true match keeps its row (|Δv| ≤ tol px, full-res).
double epipolar_ratio(const Pairs& p, const Set& a, const Set& b, float tol) {
  if (p.empty()) return 0.0;
  std::size_t ok = 0;
  for (const auto& m : p)
    if (std::fabs(a.kp[m.first * 2 + 1] - b.kp[m.second * 2 + 1]) <= tol) ++ok;
  return double(ok) / double(p.size());
}

/// Histogram of |Δv| in engine rows (0, 1, 2, 3+) as a JSON array.
std::string row_hist(const Pairs& p, const Set& a, const Set& b, float row_px) {
  std::size_t h[4] = {0, 0, 0, 0};
  for (const auto& m : p) {
    const double rows = std::fabs(a.kp[m.first * 2 + 1] - b.kp[m.second * 2 + 1]) / row_px;
    ++h[std::min<std::size_t>(3, std::size_t(std::lround(rows)))];
  }
  return "[" + std::to_string(h[0]) + ", " + std::to_string(h[1]) + ", " + std::to_string(h[2]) + ", " +
         std::to_string(h[3]) + "]";
}

/// Right = left shifted by a non-negative disparity (fixture: 0..40 px).
double disparity_ratio(const Pairs& p, const Set& a, const Set& b, float dmax) {
  if (p.empty()) return 0.0;
  std::size_t ok = 0;
  for (const auto& m : p) {
    const float d = a.kp[m.first * 2] - b.kp[m.second * 2];
    if (d >= -1.f && d <= dmax) ++ok;
  }
  return double(ok) / double(p.size());
}

okvis::xfeat::PairMatches run(okvis::xfeat::LighterGlueMatcher& lg, const Set& a, const Set& b,
                              std::uint32_t w, std::uint32_t h) {
  return lg.match(a.kp.data(), a.de.data(), a.sc.data(), a.n(), w, h,
                  b.kp.data(), b.de.data(), b.sc.data(), b.n(), w, h);
}

}  // namespace

int main(int argc, char** argv) {
  std::map<std::string, std::string> a;
  for (int i = 1; i + 1 < argc; i += 2) a[argv[i]] = argv[i + 1];
  auto get = [&](const char* k, const char* def = "") {
    auto it = a.find(k);
    return it == a.end() ? std::string(def) : it->second;
  };
  const std::string engine = get("--engine"), xfeat_engine = get("--xfeat-engine");
  const std::string left = get("--left"), right = get("--right"), json = get("--json");
  const std::string dump = get("--dump");  // per-match CSV (diagnostics)
  const int iters = std::atoi(get("--iters", "200").c_str());
  const int warmup = std::atoi(get("--warmup", "10").c_str());
  const float nn_threshold = std::strtof(get("--nn-threshold", "0.25").c_str(), nullptr);
  // px, full-res; ticket: rectified fixture → |Δy| ≤ 2 px. Needs the T-0114
  // sub-pixel "offsets" engine output: with integer keypoints at 640x384 the
  // vertical pitch on a 1280x800 image is 800/384 = 2.083 px, so one-row
  // neighbours miss a 2 px tolerance by 0.08 px (0.83 vs 0.96 measured); the
  // *_1row keys and the row histogram below make that visible either way.
  const float epi_tol = std::strtof(get("--epi-tol", "2.0").c_str(), nullptr);
  if (engine.empty() || xfeat_engine.empty() || left.empty() || right.empty() || iters <= warmup) {
    std::cerr << "usage: lighterglue_demo --engine lg.plan --xfeat-engine xfeat.plan --left L.png "
                 "--right R.png [--iters N] [--warmup N] [--nn-threshold 0.25] [--json out.json]\n";
    return 1;
  }
  cv::Mat L = cv::imread(left, cv::IMREAD_GRAYSCALE);
  cv::Mat R = cv::imread(right, cv::IMREAD_GRAYSCALE);
  if (L.empty() || R.empty() || L.size() != R.size() || !L.isContinuous() || !R.isContinuous()) {
    std::cerr << "cannot read " << left << " / " << right << " as an equal-size mono pair\n";
    return 2;
  }
  const std::uint32_t W = std::uint32_t(L.cols), H = std::uint32_t(L.rows);

  // 1. XFeat on the pair (one batch-2 enqueue, T-0111).
  okvis::xfeat::XFeatConfig xcfg;
  xcfg.engine_path = xfeat_engine;
  okvis::xfeat::XFeatFrontend fe(xcfg);
  if (!fe.engine_loaded()) { std::cerr << "xfeat engine failed to load: " << xfeat_engine << "\n"; return 3; }
  const auto tx0 = std::chrono::steady_clock::now();
  const okvis::xfeat::FrameFeatures ff = fe.extract_pair(L.data, W, R.data, W, W, H);
  const double xfeat_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tx0).count();
  const auto& sl = ff.streams.at(0);
  const auto& sr = ff.streams.at(1);

  // 2. LighterGlue on the top-K by score of each eye.
  okvis::xfeat::LighterGlueConfig cfg;
  cfg.engine_path = engine;
  okvis::xfeat::LighterGlueMatcher lg(cfg);
  if (!lg.loaded()) { std::cerr << "lighterglue engine failed to load: " << engine << "\n"; return 3; }
  const std::size_t K = lg.capacity();
  const Set A = take(sl, K), B = take(sr, K);
  okvis::xfeat::PairMatches pm;
  std::vector<double> ms;
  for (int i = 0; i < iters; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    pm = run(lg, A, B, W, H);
    const auto t1 = std::chrono::steady_clock::now();
    if (i >= warmup) ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  std::vector<double> sorted = ms;
  std::sort(sorted.begin(), sorted.end());
  const double mean = std::accumulate(ms.begin(), ms.end(), 0.0) / double(ms.size());
  const double p99 = sorted[std::min(sorted.size() - 1, std::size_t(std::ceil(0.99 * sorted.size())) - 1)];
  const double mean_score = pm.size() ? std::accumulate(pm.scores.begin(), pm.scores.end(), 0.0) / double(pm.size()) : 0.0;

  // 3. Mutual-NN cosine on the same sets (KB 02 stereo row baseline).
  const Pairs nn = mutual_nn(A, B, nn_threshold);

  // 4. Padding-mask proof: K-64 real rows padded by the matcher (score -1
  //    slots) vs the same rows + 64 garbage rows with score -1 supplied by us.
  //    Identical index sets ⇒ masked keys contributed nothing (KB 03 §padding).
  const std::size_t nReal = std::min(K - 64, std::min(A.n(), B.n()));
  const Set A0 = take(sl, nReal), B0 = take(sr, nReal);
  Set A1 = A0, B1 = B0;
  A1.append(garbage(K - nReal, W, H, 1));
  B1.append(garbage(K - nReal, W, H, 2));
  const okvis::xfeat::PairMatches r0 = run(lg, A0, B0, W, H);
  const okvis::xfeat::PairMatches r1 = run(lg, A1, B1, W, H);
  const std::set<std::pair<std::uint32_t, std::uint32_t>> s0(r0.indices.begin(), r0.indices.end());
  const std::set<std::pair<std::uint32_t, std::uint32_t>> s1(r1.indices.begin(), r1.indices.end());
  const bool ab_identical = s0 == s1 && !s0.empty();
  const bool padded = r0.masked_a > 0 && r0.masked_b > 0 && r1.masked_a == r0.masked_a && r1.masked_b == r0.masked_b;
  const int padding_masked = (ab_identical && padded && lg.io_names_match()) ? 1 : 0;

  if (!dump.empty()) {  // matcher,uL,vL,uR,vR,score — for offline Δv / score analysis
    std::ofstream f(dump);
    f << "matcher,uL,vL,uR,vR,score\n";
    for (std::size_t i = 0; i < pm.size(); ++i) {
      const auto& m = pm.indices[i];
      f << "lg," << A.kp[m.first * 2] << "," << A.kp[m.first * 2 + 1] << "," << B.kp[m.second * 2] << ","
        << B.kp[m.second * 2 + 1] << "," << pm.scores[i] << "\n";
    }
    for (const auto& m : nn)
      f << "nn," << A.kp[m.first * 2] << "," << A.kp[m.first * 2 + 1] << "," << B.kp[m.second * 2] << ","
        << B.kp[m.second * 2 + 1] << ",0\n";
  }

  std::string out = "{\n";
  auto kv = [&](const char* k, const std::string& v, bool last = false) {
    out += std::string("  \"") + k + "\": " + v + (last ? "\n" : ",\n");
  };
  kv("engine", "\"" + engine + "\"");
  kv("xfeat_engine", "\"" + xfeat_engine + "\"");
  kv("left", "\"" + left + "\"");
  kv("right", "\"" + right + "\"");
  kv("image_width", std::to_string(W));
  kv("image_height", std::to_string(H));
  kv("k", std::to_string(K));
  kv("io_names_match", std::to_string(int(lg.io_names_match())));
  kv("xfeat_pair_ms", std::to_string(xfeat_ms));
  kv("xfeat_subpixel", std::to_string(int(fe.has_offsets())));
  kv("left_keypoints", std::to_string(sl.size()));
  kv("right_keypoints", std::to_string(sr.size()));
  kv("staged_left", std::to_string(pm.staged_a));
  kv("staged_right", std::to_string(pm.staged_b));
  kv("masked_left", std::to_string(pm.masked_a));
  kv("masked_right", std::to_string(pm.masked_b));
  kv("matches", std::to_string(pm.size()));
  std::uint32_t ew = 0, eh = 0;
  fe.input_dims(ew, eh);
  const float row_px = float(H) / float(eh);  // full-res px per engine row
  kv("epipolar_tol_px", std::to_string(epi_tol));
  kv("engine_row_px", std::to_string(row_px));
  kv("epipolar_inlier_ratio", std::to_string(epipolar_ratio(pm.indices, A, B, epi_tol)));
  kv("epipolar_inlier_ratio_1row", std::to_string(epipolar_ratio(pm.indices, A, B, row_px + 0.01f)));
  kv("epipolar_row_hist", row_hist(pm.indices, A, B, row_px));
  kv("disparity_inlier_ratio", std::to_string(disparity_ratio(pm.indices, A, B, 45.f)));
  kv("mean_match_score", std::to_string(mean_score));
  kv("min_score", std::to_string(cfg.min_score));
  kv("nn_threshold", std::to_string(nn_threshold));
  kv("nn_matches", std::to_string(nn.size()));
  kv("nn_epipolar_inlier_ratio", std::to_string(epipolar_ratio(nn, A, B, epi_tol)));
  kv("nn_epipolar_inlier_ratio_1row", std::to_string(epipolar_ratio(nn, A, B, row_px + 0.01f)));
  kv("nn_epipolar_row_hist", row_hist(nn, A, B, row_px));
  kv("mask_real_rows", std::to_string(nReal));
  kv("mask_run_padded_matches", std::to_string(r0.size()));
  kv("mask_run_garbage_matches", std::to_string(r1.size()));
  kv("mask_ab_identical", std::to_string(int(ab_identical)));
  kv("padding_masked", std::to_string(padding_masked));
  kv("iters", std::to_string(iters));
  kv("warmup", std::to_string(warmup));
  kv("mean_ms", std::to_string(mean));
  kv("p99_ms", std::to_string(p99));
  kv("min_ms", std::to_string(sorted.front()));
  kv("max_ms", std::to_string(sorted.back()), true);
  out += "}\n";
  std::cout << out;
  if (!json.empty()) {
    std::ofstream f(json);
    if (!f) { std::cerr << "cannot write " << json << "\n"; return 4; }
    f << out;
  }
  return 0;
}
