// CPU-only gtest for the XFeat readback helpers (T-0111): padding strip,
// full-res coordinate scale-back, unit-norm check. Runs on host, qemu and device.
#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "okvis/xfeat/FrameFeaturesUtil.hpp"

using namespace okvis::xfeat;

namespace {
// Synthetic engine tensors for one batch slot: K rows, the last `pad` are
// padding (score -1, garbage coords/descriptors), the rest unit-norm.
struct Tensors {
  std::vector<std::int32_t> kp;
  std::vector<float> sc, de;
};
Tensors make(std::uint32_t K, std::uint32_t pad, unsigned seed = 1) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> n;
  Tensors t;
  t.kp.resize(K * 2);
  t.sc.resize(K);
  t.de.resize(std::size_t(K) * kDescriptorDim);
  for (std::uint32_t i = 0; i < K; ++i) {
    t.kp[2 * i] = int(i % 640);
    t.kp[2 * i + 1] = int(i % 384);
    double acc = 0;
    for (std::uint32_t k = 0; k < kDescriptorDim; ++k) {
      float v = n(rng);
      t.de[i * kDescriptorDim + k] = v;
      acc += double(v) * v;
    }
    for (std::uint32_t k = 0; k < kDescriptorDim; ++k) t.de[i * kDescriptorDim + k] /= float(std::sqrt(acc));
    t.sc[i] = (i + pad < K) ? 0.1f + 0.8f * float(i) / K : -1.f;
  }
  return t;
}
}  // namespace

TEST(FrameFeatures, PaddingRowsAreStripped) {
  const std::uint32_t K = 1024, pad = 200;
  Tensors t = make(K, pad);
  StreamFeatures sf;
  const auto stripped = assemble_stream(t.kp.data(), t.sc.data(), t.de.data(), K, {1.f, 1.f}, 0.f, sf);
  EXPECT_EQ(stripped, pad);
  EXPECT_EQ(sf.padding_rows, pad);
  EXPECT_EQ(sf.size(), K - pad);
  EXPECT_EQ(sf.scores.size(), K - pad);
  EXPECT_EQ(sf.descriptors.size(), std::size_t(K - pad) * kDescriptorDim);
  for (float s : sf.scores) EXPECT_GE(s, 0.f);
  // Row order preserved: descriptor row i is source row i.
  for (std::uint32_t k = 0; k < kDescriptorDim; ++k) EXPECT_EQ(sf.descriptors[k], t.de[k]);
}

TEST(FrameFeatures, ScoreThresholdAlsoDrops) {
  Tensors t = make(100, 10);
  StreamFeatures sf;
  assemble_stream(t.kp.data(), t.sc.data(), t.de.data(), 100, {1.f, 1.f}, 0.5f, sf);
  EXPECT_EQ(sf.padding_rows, 10u);
  for (float s : sf.scores) EXPECT_GE(s, 0.5f);
  EXPECT_LT(sf.size(), 90u);
  EXPECT_GT(sf.size(), 0u);
}

TEST(FrameFeatures, NoPaddingIsFine) {
  Tensors t = make(64, 0);
  StreamFeatures sf;
  EXPECT_EQ(assemble_stream(t.kp.data(), t.sc.data(), t.de.data(), 64, {1.f, 1.f}, 0.f, sf), 0u);
  EXPECT_EQ(sf.size(), 64u);
}

TEST(FrameFeatures, ScaleBackToFullRes) {
  // 1280x800 → 640x384: sx = 2, sy = 800/384 (KB 03 §resolution).
  const ResizeScale s = resize_scale(1280, 800, 640, 384);
  EXPECT_FLOAT_EQ(s.sx, 2.f);
  EXPECT_NEAR(s.sy, 2.083333f, 1e-6f);
  // Pixel-centre convention: engine (0,0) covers full-res rows/cols 0..1 → 0.5.
  Keypoint k0 = to_full_res(0, 0, s);
  EXPECT_FLOAT_EQ(k0.u, 0.5f);
  EXPECT_NEAR(k0.v, 0.541667f, 1e-5f);
  // Last engine pixel lands inside the image, near the far edge.
  Keypoint k1 = to_full_res(639, 383, s);
  EXPECT_FLOAT_EQ(k1.u, 1278.5f);
  EXPECT_NEAR(k1.v, 798.4583f, 1e-3f);
  EXPECT_LT(k1.u, 1280.f);
  EXPECT_LT(k1.v, 800.f);
  // Identity scale is exact.
  Keypoint k2 = to_full_res(17, 23, {1.f, 1.f});
  EXPECT_FLOAT_EQ(k2.u, 17.f);
  EXPECT_FLOAT_EQ(k2.v, 23.f);
  // Through assemble_stream.
  std::int32_t kp[2] = {100, 50};
  float sc[1] = {0.9f};
  std::vector<float> de(kDescriptorDim, 0.f);
  de[0] = 1.f;
  StreamFeatures sf;
  assemble_stream(kp, sc, de.data(), 1, s, 0.f, sf);
  ASSERT_EQ(sf.size(), 1u);
  EXPECT_FLOAT_EQ(sf.keypoints_px[0].u, 200.5f);
  EXPECT_NEAR(sf.keypoints_px[0].v, 104.7083f, 1e-3f);
}

TEST(FrameFeatures, UnitNormCheck) {
  Tensors t = make(50, 0);
  EXPECT_LT(max_unit_norm_deviation(t.de), 1e-5f);
  EXPECT_FLOAT_EQ(max_unit_norm_deviation({}), 0.f);
  for (std::uint32_t k = 0; k < kDescriptorDim; ++k) t.de[3 * kDescriptorDim + k] *= 3.f;  // break one row
  EXPECT_GT(max_unit_norm_deviation(t.de), 0.1f);
  // Padding rows carry garbage norms but are stripped before the check.
  Tensors p = make(50, 5);
  for (std::uint32_t k = 0; k < kDescriptorDim; ++k) p.de[49 * kDescriptorDim + k] = 7.f;
  StreamFeatures sf;
  assemble_stream(p.kp.data(), p.sc.data(), p.de.data(), 50, {1.f, 1.f}, 0.f, sf);
  EXPECT_LT(max_unit_norm_deviation(sf.descriptors), 1e-5f);
}
