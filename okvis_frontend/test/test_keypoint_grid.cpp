/**
 * @file test_keypoint_grid.cpp
 * @brief T-0128: KeypointGrid candidate bucketing for Frontend::matchToMapByThread.
 *
 * Synthetic keypoints / landmark projections on a 512x512 image (TUM-VI XFeat size),
 * gate radius 14.46 px (3 + f*0.06 at f = 191, the IMU case in matchToMap):
 *  - the grid candidates are a superset of the keypoints inside the gate (incl.
 *    projections up to the gate outside the image and keypoints on the border);
 *  - best cosine match per keypoint through the grid == full scan, bit for bit.
 */
#include <gtest/gtest.h>

#include <random>
#include <set>
#include <vector>

#include <okvis/DescriptorDistance.hpp>
#include <okvis/KeypointGrid.hpp>

namespace {
constexpr double kW = 512.0, kH = 512.0, kR = 3.0 + 191.0 * 0.06;

Eigen::Matrix2Xd randomKeypoints(std::mt19937& rng, size_t n) {
  std::uniform_real_distribution<double> u(0.0, kW), v(0.0, kH);
  Eigen::Matrix2Xd kps(2, n);
  for (size_t k = 0; k < n; ++k) kps.col(k) << u(rng), v(rng);
  kps.col(0) << 0.0, 0.0;           // corners / border exactly
  kps.col(1) << kW - 1e-9, kH - 1e-9;
  return kps;
}

std::vector<float> unitDescriptors(std::mt19937& rng, size_t n) {
  std::normal_distribution<float> g(0.f, 1.f);
  std::vector<float> d(n * okvis::kFloatDescriptorDim);
  for (size_t i = 0; i < n; ++i) {
    Eigen::Map<Eigen::VectorXf> row(&d[i * okvis::kFloatDescriptorDim], okvis::kFloatDescriptorDim);
    for (int j = 0; j < okvis::kFloatDescriptorDim; ++j) row[j] = g(rng);
    row.normalize();
  }
  return d;
}
}  // namespace

TEST(KeypointGrid, CandidatesAreSupersetOfGate) {
  std::mt19937 rng(7);
  const size_t n = 700;
  const Eigen::Matrix2Xd kps = randomKeypoints(rng, n);
  std::vector<bool> use(n, true);
  for (size_t k = 5; k < n; k += 11) use[k] = false;  // already-matched keypoints
  // thread segment like matchToMapByThread: [begin, end)
  const size_t begin = 100, end = 600;
  const okvis::KeypointGrid grid(kps, begin, end, use, kW, kH, kR);
  std::uniform_real_distribution<double> q(-kR, kW + kR);
  size_t visited = 0, inside = 0;
  for (int t = 0; t < 5000; ++t) {
    const Eigen::Vector2d p(q(rng), q(rng));
    std::set<size_t> cand;
    grid.forEachNear(p, kR, [&](size_t k) {
      EXPECT_TRUE(k >= begin && k < end && use[k]);
      EXPECT_TRUE(cand.insert(k).second) << "keypoint visited twice";
    });
    visited += cand.size();
    for (size_t k = begin; k < end; ++k) {
      if (use[k] && (kps.col(k) - p).squaredNorm() <= kR * kR) {
        ++inside;
        EXPECT_TRUE(cand.count(k)) << "gate keypoint " << k << " missed for p=" << p.transpose();
      }
    }
  }
  // the point of the grid: far fewer visits than the full segment (500 per query)
  EXPECT_LT(double(visited) / 5000.0, 0.1 * double(end - begin));
  EXPECT_GT(inside, 0u);
}

TEST(KeypointGrid, BestCosineMatchEqualsFullScan) {
  std::mt19937 rng(11);
  const size_t nk = 700, nl = 1500;
  const Eigen::Matrix2Xd kps = randomKeypoints(rng, nk);
  std::vector<float> kd = unitDescriptors(rng, nk), ld = unitDescriptors(rng, nl);
  // landmarks: half are true re-observations (projection near a keypoint, noisy descriptor)
  std::normal_distribution<double> px(0.0, 4.0);
  std::normal_distribution<float> dn(0.f, 0.05f);
  Eigen::Matrix2Xd proj(2, nl);
  std::uniform_real_distribution<double> q(-kR, kW + kR);
  for (size_t l = 0; l < nl; ++l) {
    if (l % 2 == 0) {
      const size_t k = l % nk;
      proj.col(l) = kps.col(k) + Eigen::Vector2d(px(rng), px(rng));
      Eigen::Map<Eigen::VectorXf> row(&ld[l * okvis::kFloatDescriptorDim], okvis::kFloatDescriptorDim);
      for (int j = 0; j < okvis::kFloatDescriptorDim; ++j)
        row[j] = kd[k * okvis::kFloatDescriptorDim + j] + dn(rng);
      row.normalize();
    } else {
      proj.col(l) << q(rng), q(rng);
    }
  }
  const std::vector<bool> use(nk, true);
  const double threshold = 0.25;  // matching_threshold (cosine distance)
  auto score = [&](size_t k, size_t l, std::vector<double>& dist, std::vector<long>& id) {
    const Eigen::Vector2d r = proj.col(l) - kps.col(k);
    if (r.dot(r) > kR * kR) return;
    const double d = okvis::cosineDistance(
        reinterpret_cast<const unsigned char*>(&kd[k * okvis::kFloatDescriptorDim]),
        reinterpret_cast<const unsigned char*>(&ld[l * okvis::kFloatDescriptorDim]));
    if (d < dist[k]) { dist[k] = d; id[k] = long(l); }
  };
  std::vector<double> distFull(nk, threshold), distGrid(nk, threshold);
  std::vector<long> idFull(nk, -1), idGrid(nk, -1);
  for (size_t l = 0; l < nl; ++l)
    for (size_t k = 0; k < nk; ++k) score(k, l, distFull, idFull);
  const okvis::KeypointGrid grid(kps, 0, nk, use, kW, kH, kR);
  for (size_t l = 0; l < nl; ++l)
    grid.forEachNear(proj.col(l), kR, [&](size_t k) { score(k, l, distGrid, idGrid); });
  size_t matched = 0;
  for (size_t k = 0; k < nk; ++k) {
    EXPECT_EQ(idFull[k], idGrid[k]) << "keypoint " << k;
    EXPECT_EQ(distFull[k], distGrid[k]) << "keypoint " << k;  // identical scores, not just close
    matched += idFull[k] >= 0;
  }
  EXPECT_GE(matched, 300u);  // planted re-observations hit the 350 even keypoints
}
