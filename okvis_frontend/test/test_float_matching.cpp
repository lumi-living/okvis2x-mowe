/**
 * @file test_float_matching.cpp
 * @brief T-0112: float 64-D / cosine descriptor seam vs. BRISK / Hamming.
 *
 * Pure CPU (no TensorRT, no camera): synthetic unit-norm descriptor sets with
 * known correspondences + distractors, stored in okvis::MultiFrame exactly the
 * way detectAndDescribeXFeat() stores them (CV_32FC1 K×64 via
 * resetDescriptors), read back through keypointDescriptor() (the .step-based
 * row pointer the matching loops use) and ranked with the Frontend's
 * descriptorDist() seam (okvis/DescriptorDistance.hpp). Runs under
 * qemu-aarch64 and on the device.
 */
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <okvis/DescriptorDistance.hpp>
#include <okvis/Frontend.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/cameras/NCameraSystem.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/NoDistortion.hpp>

namespace {

// Minimal 2-camera system so MultiFrame can be constructed like in the estimator.
okvis::cameras::NCameraSystem makeStereo() {
  okvis::cameras::NCameraSystem cams;
  for (int i = 0; i < 2; ++i) {
    std::shared_ptr<const okvis::kinematics::Transformation> T_SC(
        new okvis::kinematics::Transformation(Eigen::Vector3d(0.07 * i, 0, 0),
                                              Eigen::Quaterniond::Identity()));
    std::shared_ptr<const okvis::cameras::CameraBase> cam(
        new okvis::cameras::PinholeCamera<okvis::cameras::NoDistortion>(
            640, 384, 400.0, 400.0, 320.0, 192.0, okvis::cameras::NoDistortion()));
    cams.addCamera(T_SC, cam, okvis::cameras::NCameraSystem::DistortionType::NoDistortion,
                   /*computeOverlaps=*/false);
  }
  return cams;
}

// K unit-norm 64-D rows (rows of a CV_32FC1 Mat), gaussian then normalised.
cv::Mat randomUnitDescriptors(int K, std::mt19937& rng) {
  std::normal_distribution<float> n(0.f, 1.f);
  cv::Mat d(K, okvis::kFloatDescriptorDim, CV_32FC1);
  for (int k = 0; k < K; ++k) {
    float* row = d.ptr<float>(k);
    float norm2 = 0.f;
    for (int j = 0; j < okvis::kFloatDescriptorDim; ++j) { row[j] = n(rng); norm2 += row[j] * row[j]; }
    const float inv = 1.f / std::sqrt(norm2);
    for (int j = 0; j < okvis::kFloatDescriptorDim; ++j) row[j] *= inv;
  }
  return d;
}

// Perturb a unit row with gaussian noise of std sigma per dim, renormalise.
void perturbRow(const float* in, float* out, float sigma, std::mt19937& rng) {
  std::normal_distribution<float> n(0.f, sigma);
  float norm2 = 0.f;
  for (int j = 0; j < okvis::kFloatDescriptorDim; ++j) { out[j] = in[j] + n(rng); norm2 += out[j] * out[j]; }
  const float inv = 1.f / std::sqrt(norm2);
  for (int j = 0; j < okvis::kFloatDescriptorDim; ++j) out[j] *= inv;
}

std::vector<cv::KeyPoint> dummyKeypoints(int K) {
  std::vector<cv::KeyPoint> kps;
  for (int k = 0; k < K; ++k) kps.emplace_back(float(10 + k % 600), float(10 + k / 600), 16.f);
  return kps;
}

// Mutual nearest neighbour under the frontend's metric, the way the Frontend
// loops do it: raw row pointers from keypointDescriptor(), distance via
// descriptorDist(), accept strictly below the matching threshold.
std::vector<int> mutualNN(const okvis::Frontend& fe, const okvis::MultiFrame& a, size_t imA,
                          const okvis::MultiFrame& b, size_t imB, double threshold) {
  const size_t nA = a.numKeypoints(imA), nB = b.numKeypoints(imB);
  std::vector<int> bestForA(nA, -1), bestForB(nB, -1);
  std::vector<double> distA(nA, threshold), distB(nB, threshold);
  for (size_t i = 0; i < nA; ++i) {
    const unsigned char* da = a.keypointDescriptor(imA, i);
    for (size_t j = 0; j < nB; ++j) {
      const double d = fe.descriptorDist(da, b.keypointDescriptor(imB, j));
      if (d < distA[i]) { distA[i] = d; bestForA[i] = int(j); }
      if (d < distB[j]) { distB[j] = d; bestForB[j] = int(i); }
    }
  }
  std::vector<int> out(nA, -1);
  for (size_t i = 0; i < nA; ++i)
    if (bestForA[i] >= 0 && bestForB[size_t(bestForA[i])] == int(i)) out[i] = bestForA[i];
  return out;
}

}  // namespace

TEST(FloatDescriptors, FrameStoresFloatRowsWithStepNotCols) {
  std::mt19937 rng(1);
  const int K = 37;
  auto cams = makeStereo();
  okvis::MultiFrame mf(cams, okvis::Time(0.0), 1);
  // Engine output has padded slots (score -1); detectAndDescribeXFeat drops them
  // before resetDescriptors, so numKeypoints() must equal the un-padded count.
  cv::Mat d = randomUnitDescriptors(K, rng);
  mf.resetKeypoints(0, dummyKeypoints(K));
  mf.resetDescriptors(0, d);
  EXPECT_EQ(mf.numKeypoints(0), size_t(K));
  EXPECT_EQ(mf.descriptors(0).type(), CV_32FC1);
  EXPECT_EQ(mf.descriptors(0).cols, okvis::kFloatDescriptorDim);
  EXPECT_EQ(mf.descriptors(0).step, okvis::kFloatDescriptorBytes);  // 256 B, not 64
  for (int k = 0; k < K; ++k) {
    EXPECT_EQ(mf.keypointDescriptor(0, size_t(k)),
              reinterpret_cast<const unsigned char*>(d.ptr<float>(k)));
  }
  EXPECT_TRUE(okvis::DescriptorMetric::forDescriptors(mf.descriptors(0)).isFloat);
  EXPECT_EQ(okvis::DescriptorMetric::forDescriptors(mf.descriptors(0)).bytes(),
            okvis::kFloatDescriptorBytes);
}

TEST(FloatDescriptors, CosineDistanceIdentities) {
  std::mt19937 rng(2);
  cv::Mat d = randomUnitDescriptors(2, rng);
  const unsigned char* a = d.ptr<unsigned char>(0);
  const unsigned char* b = d.ptr<unsigned char>(1);
  EXPECT_NEAR(okvis::cosineDistance(a, a), 0.0, 1e-5);
  // 1 - a.b equals |a-b|^2 / 2 for unit vectors.
  Eigen::Map<const Eigen::VectorXf> fa(d.ptr<float>(0), 64), fb(d.ptr<float>(1), 64);
  EXPECT_NEAR(okvis::cosineDistance(a, b), 0.5 * (fa - fb).squaredNorm(), 1e-4);
  // antipodal -> 2
  cv::Mat neg = -d.row(0);
  EXPECT_NEAR(okvis::cosineDistance(a, neg.ptr<unsigned char>(0)), 2.0, 1e-5);
}

TEST(FloatDescriptors, CosineMutualNNPrecisionRecall) {
  std::mt19937 rng(3);
  const int K = 400, distractors = 200;
  auto cams = makeStereo();
  okvis::MultiFrame f0(cams, okvis::Time(0.0), 1), f1(cams, okvis::Time(0.05), 2);

  cv::Mat d0 = randomUnitDescriptors(K, rng);
  // frame 1: noisy copies of the K true ones (shuffled) + unrelated distractors
  std::vector<int> perm(K);
  for (int i = 0; i < K; ++i) perm[i] = i;
  std::shuffle(perm.begin(), perm.end(), rng);
  cv::Mat d1(K + distractors, okvis::kFloatDescriptorDim, CV_32FC1);
  std::vector<int> truth(K);  // truth[k0] = k1
  for (int k1 = 0; k1 < K; ++k1) {
    perturbRow(d0.ptr<float>(perm[k1]), d1.ptr<float>(k1), 0.04f, rng);  // ~cos dist 0.05
    truth[perm[k1]] = k1;
  }
  cv::Mat dist = randomUnitDescriptors(distractors, rng);
  dist.copyTo(d1.rowRange(K, K + distractors));

  f0.resetKeypoints(0, dummyKeypoints(K));               f0.resetDescriptors(0, d0);
  f1.resetKeypoints(0, dummyKeypoints(K + distractors)); f1.resetDescriptors(0, d1);

  okvis::Frontend fe(2, "/nonexistent/voc/dir");  // lazy DBoW: must not touch the file
  fe.setFloatDescriptors(true);
  EXPECT_TRUE(fe.floatDescriptors());
  EXPECT_EQ(fe.descriptorBytes(), okvis::kFloatDescriptorBytes);
  const double threshold = 0.25;  // config/mowe/okvis2-xfeat.yaml matching_threshold

  const std::vector<int> m = mutualNN(fe, f0, 0, f1, 0, threshold);
  int tp = 0, fp = 0;
  for (int k0 = 0; k0 < K; ++k0) {
    if (m[k0] < 0) continue;
    if (m[k0] == truth[k0]) ++tp; else ++fp;
  }
  const double precision = double(tp) / double(tp + fp);
  const double recall = double(tp) / double(K);
  std::cout << "cosine mutual-NN: tp=" << tp << " fp=" << fp << " precision=" << precision
            << " recall=" << recall << std::endl;
  EXPECT_GE(precision, 0.95);
  EXPECT_GE(recall, 0.90);
}

TEST(FloatDescriptors, DistractorsRejectedByThreshold) {
  // Random unit vectors in 64-D have cos ~ N(0, 1/8): a 0.25 cosine-distance
  // gate (cos > 0.75) must reject essentially all unrelated pairs.
  std::mt19937 rng(4);
  cv::Mat a = randomUnitDescriptors(300, rng), b = randomUnitDescriptors(300, rng);
  okvis::DescriptorMetric cosine{true};
  int accepted = 0;
  for (int i = 0; i < 300; ++i)
    for (int j = 0; j < 300; ++j)
      if (cosine(a.ptr<unsigned char>(i), b.ptr<unsigned char>(j)) < 0.25) ++accepted;
  EXPECT_LE(accepted, 2) << "of 90000 random pairs";
}

TEST(BriskDescriptors, HammingPathUnchanged) {
  // BRISK/Hamming regression: 48-byte binary rows, distance == flipped bits,
  // and a frontend with float descriptors OFF dispatches to Hamming.
  std::mt19937 rng(5);
  std::uniform_int_distribution<int> byte(0, 255);
  const int K = 200;
  cv::Mat d0(K, int(okvis::kBriskDescriptorBytes), CV_8UC1);
  for (int k = 0; k < K; ++k)
    for (int c = 0; c < int(okvis::kBriskDescriptorBytes); ++c) d0.at<uchar>(k, c) = uchar(byte(rng));
  // frame 1: each row with exactly `flips` random bits flipped
  cv::Mat d1 = d0.clone();
  const int flips = 20;
  std::uniform_int_distribution<int> bit(0, 48 * 8 - 1);
  for (int k = 0; k < K; ++k) {
    std::vector<bool> done(48 * 8, false);
    for (int f = 0; f < flips;) {
      int bi = bit(rng);
      if (done[bi]) continue;
      done[bi] = true; ++f;
      d1.at<uchar>(k, bi / 8) ^= uchar(1u << (bi % 8));
    }
  }
  okvis::DescriptorMetric hamming{false};
  EXPECT_EQ(hamming.bytes(), okvis::kBriskDescriptorBytes);
  for (int k = 0; k < K; ++k) {
    EXPECT_DOUBLE_EQ(hamming(d0.ptr<uchar>(k), d1.ptr<uchar>(k)), double(flips));
    EXPECT_DOUBLE_EQ(hamming(d0.ptr<uchar>(k), d0.ptr<uchar>(k)), 0.0);
  }
  EXPECT_FALSE(okvis::DescriptorMetric::forDescriptors(d0).isFloat);

  auto cams = makeStereo();
  okvis::MultiFrame f0(cams, okvis::Time(0.0), 1), f1(cams, okvis::Time(0.05), 2);
  f0.resetKeypoints(0, dummyKeypoints(K)); f0.resetDescriptors(0, d0);
  f1.resetKeypoints(0, dummyKeypoints(K)); f1.resetDescriptors(0, d1);
  EXPECT_EQ(f0.descriptors(0).step, okvis::kBriskDescriptorBytes);  // .step == .cols for CV_8U

  okvis::Frontend fe(2, "/nonexistent/voc/dir");
  EXPECT_FALSE(fe.floatDescriptors());  // default: BRISK
  EXPECT_EQ(fe.descriptorBytes(), okvis::kBriskDescriptorBytes);
  const std::vector<int> m = mutualNN(fe, f0, 0, f1, 0, 60.0);  // okvis2.yaml matching_threshold
  int correct = 0;
  for (int k = 0; k < K; ++k) if (m[k] == k) ++correct;
  EXPECT_EQ(correct, K);  // 20 flipped bits vs ~192 expected for random rows: all found
}

TEST(BriskDescriptors, LazyVocabularyNotLoadedUntilNeeded) {
  // Constructing a Frontend with a bogus vocabulary dir must not throw (T-0112
  // lazy DBoW): the XFeat path never needs small_voc.yml.gz.
  EXPECT_NO_THROW({
    okvis::Frontend fe(2, "/nonexistent/voc/dir");
    fe.setFloatDescriptors(true);
    fe.clear();  // clear() must tolerate the never-loaded database
  });
}
