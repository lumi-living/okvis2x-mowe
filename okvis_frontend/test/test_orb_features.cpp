/**
 * @file test_orb_features.cpp
 * @brief T-0134: cv::ORB behind the BRISK seam (okvis/OrbFeatures.hpp).
 *
 * Deterministic synthetic fixture frame (seeded blobs, blurred), pure CPU.
 * - detector: orb → N <= max keypoints, 32-byte ORB rows (bytes 32..47 of the
 *   48-byte BRISK row are zero, and equal to cv::ORB's own descriptors),
 *   Hamming self-distance 0, sizes on BRISK's scale.
 * - detector: brisk → descriptors byte-identical to before T-0134 (golden
 *   FNV-1a hash) and after an orb → brisk round trip.
 */
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>

#include <opencv2/imgproc.hpp>

#include <okvis/DescriptorDistance.hpp>
#include <okvis/Frontend.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/OrbFeatures.hpp>
#include <okvis/cameras/NCameraSystem.hpp>
#include <okvis/cameras/NoDistortion.hpp>
#include <okvis/cameras/PinholeCamera.hpp>

namespace {

constexpr int kW = 512, kH = 512, kMaxKp = 300;

cv::Mat fixtureImage() {
  cv::Mat img(kH, kW, CV_8UC1, cv::Scalar(90));
  cv::RNG rng(1234);
  for (int i = 0; i < 400; ++i) {
    const cv::Point c(rng.uniform(0, kW), rng.uniform(0, kH));
    if (i % 2) {
      cv::circle(img, c, rng.uniform(3, 18), cv::Scalar(rng.uniform(0, 256)), cv::FILLED);
    } else {
      cv::rectangle(img, c, c + cv::Point(rng.uniform(4, 30), rng.uniform(4, 30)),
                    cv::Scalar(rng.uniform(0, 256)), cv::FILLED);
    }
  }
  cv::GaussianBlur(img, img, cv::Size(3, 3), 0.8);
  return img;
}

okvis::cameras::NCameraSystem makeCams() {
  okvis::cameras::NCameraSystem cams;
  for (int i = 0; i < 2; ++i) {
    auto cam = std::make_shared<okvis::cameras::PinholeCamera<okvis::cameras::NoDistortion>>(
        kW, kH, 190.0, 190.0, 255.0, 256.0, okvis::cameras::NoDistortion());
    cam->initialiseCameraAwarenessMaps();  // BRISK camera-aware extraction
    cams.addCamera(std::make_shared<okvis::kinematics::Transformation>(
                       Eigen::Vector3d(0.1 * i, 0, 0), Eigen::Quaterniond::Identity()),
                   cam, okvis::cameras::NCameraSystem::DistortionType::NoDistortion,
                   /*computeOverlaps=*/false);
  }
  return cams;
}

// Run the Frontend's detect+describe on camera 0 of the fixture frame.
std::shared_ptr<okvis::MultiFrame> detect(okvis::Frontend& fe) {
  auto mf = std::make_shared<okvis::MultiFrame>(makeCams(), okvis::Time(1.0), 1);
  mf->setImage(0, fixtureImage());
  fe.detectAndDescribe(0, mf, okvis::kinematics::Transformation(), nullptr);
  return mf;
}

uint64_t fnv(const cv::Mat& m) {
  uint64_t h = 1469598103934665603ull;
  for (int r = 0; r < m.rows; ++r)
    for (int c = 0; c < m.cols * int(m.elemSize()); ++c) {
      h ^= m.ptr<uchar>(r)[c];
      h *= 1099511628211ull;
    }
  return h;
}

// okvis2-brisk-vpr.yaml detection settings, max keypoints lowered for the fixture.
void configure(okvis::Frontend& fe) {
  fe.setBriskDetectionOctaves(0);
  fe.setBriskDetectionThreshold(34.0);
  fe.setBriskDetectionAbsoluteThreshold(200.0);
  fe.setBriskDetectionMaximumKeypoints(kMaxKp);
}

}  // namespace

TEST(OrbFeatures, DetectDescribeShapeAndPadding) {
  okvis::Frontend fe(2, "/nonexistent/voc/dir");
  configure(fe);
  fe.setDetector("orb");
  ASSERT_TRUE(fe.usingOrb());
  EXPECT_EQ(fe.descriptorBytes(), okvis::kBriskDescriptorBytes);  // binary Hamming path
  auto mf = detect(fe);
  const size_t n = mf->numKeypoints(0);
  EXPECT_GT(n, 100u);
  EXPECT_LE(n, size_t(kMaxKp));
  const cv::Mat d = mf->descriptors(0);
  ASSERT_EQ(size_t(d.rows), n);
  ASSERT_EQ(d.cols, int(okvis::kBriskDescriptorBytes));
  ASSERT_EQ(d.type(), CV_8UC1);
  okvis::DescriptorMetric hamming{false};
  for (int k = 0; k < d.rows; ++k) {
    EXPECT_EQ(cv::countNonZero(d.row(k).colRange(okvis::kOrbDescriptorBytes, d.cols)), 0);
    EXPECT_DOUBLE_EQ(hamming(d.ptr<uchar>(k), d.ptr<uchar>(k)), 0.0);
    double size = 0.0;
    mf->getKeypointSize(0, size_t(k), size);
    EXPECT_GE(size, 12.0);  // 12 * 1.2^octave: sigma 0.8 px at full resolution
  }

  // The 32 leading bytes are exactly cv::ORB's descriptors for those keypoints.
  std::vector<cv::KeyPoint> kps;
  cv::Ptr<cv::ORB> orb = cv::ORB::create(kMaxKp, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 34);
  cv::Mat d32;
  orb->detectAndCompute(fixtureImage(), cv::noArray(), kps, d32);
  ASSERT_EQ(d32.cols, okvis::kOrbDescriptorBytes);
  ASSERT_EQ(size_t(d32.rows), n);
  EXPECT_EQ(cv::norm(d32, d.colRange(0, okvis::kOrbDescriptorBytes), cv::NORM_HAMMING), 0.0);
}

TEST(OrbFeatures, UnknownDetectorThrows) {
  okvis::Frontend fe(2, "/nonexistent/voc/dir");
  configure(fe);
  EXPECT_ANY_THROW(fe.setDetector("sift"));
}

TEST(OrbFeatures, BriskUnchanged) {
  okvis::Frontend fe(2, "/nonexistent/voc/dir");
  configure(fe);
  EXPECT_FALSE(fe.usingOrb());  // default
  const cv::Mat d0 = detect(fe)->descriptors(0).clone();
  ASSERT_GT(d0.rows, 50);
  ASSERT_EQ(d0.cols, int(okvis::kBriskDescriptorBytes));
  // Golden: BRISK descriptors on the fixture frame, recorded from the unmodified
  // BRISK path (T-0134 does not touch it).
  EXPECT_EQ(fnv(d0), 0x26b3ca7b35d33953ull) << std::hex << "fnv 0x" << fnv(d0) << std::dec << " rows " << d0.rows;
  fe.setDetector("orb");
  fe.setDetector("brisk");
  const cv::Mat d1 = detect(fe)->descriptors(0);
  ASSERT_EQ(d1.size(), d0.size());
  EXPECT_EQ(cv::norm(d0, d1, cv::NORM_HAMMING), 0.0);
}
