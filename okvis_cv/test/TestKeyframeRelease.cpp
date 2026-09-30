/**
 * @file TestKeyframeRelease.cpp
 * @brief Mow-e T-0142: retention policy of a keyframe once it becomes a pose-graph frame
 *        (okvis2 45dade8, ViSlamBackend::applyStrategy): images and depth images are released;
 *        keypoints, descriptors and landmark ids are KEPT — loop-closure verification
 *        (Frontend::verifyRecognisedPlace) and the .mowemap writer read them from the
 *        pose-graph MultiFrame, and TwoPoseGraphError revival addresses observations by
 *        keypoint index, so they cannot be compacted in place.
 */
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "okvis/cameras/PinholeCamera.hpp"
#include "okvis/cameras/NoDistortion.hpp"
#include "okvis/MultiFrame.hpp"

TEST(KeyframeRelease, imagesReleasedFeaturesKept) {
  okvis::cameras::NCameraSystem nCameraSystem;
  okvis::cameras::NCameraSystem::CameraType cameraType;
  cameraType.isColour = false;
  for (size_t i = 0; i < 2; ++i) {
    nCameraSystem.addCamera(
        std::make_shared<okvis::kinematics::Transformation>(Eigen::Vector3d(0.05 * double(i), 0, 0),
                                                            Eigen::Quaterniond(1, 0, 0, 0)),
        okvis::cameras::PinholeCamera<okvis::cameras::NoDistortion>::createTestObject(),
        okvis::cameras::NCameraSystem::NoDistortion, true, cameraType);
  }
  okvis::MultiFrame mf(nCameraSystem, okvis::Time(1.0), 7);
  std::vector<cv::Mat> held;  // an outside holder must keep its own copy alive
  for (size_t c = 0; c < 2; ++c) {
    const cv::Mat img(400, 640, CV_8UC1, cv::Scalar(100 + int(c)));
    held.push_back(img);
    mf.setImage(c, img);
    mf.setDepthImage(c, cv::Mat(400, 640, CV_32FC1, cv::Scalar(1.0f)));
    std::vector<cv::KeyPoint> kps{cv::KeyPoint(10.f, 20.f, 8.f), cv::KeyPoint(30.f, 40.f, 8.f),
                                  cv::KeyPoint(50.f, 60.f, 8.f)};
    ASSERT_TRUE(mf.resetKeypoints(c, kps));
    cv::Mat desc(3, 48, CV_8UC1, cv::Scalar(uchar(7 + c)));
    ASSERT_TRUE(mf.resetDescriptors(c, desc));
    ASSERT_TRUE(mf.setLandmarkId(c, 1, 42 + c));
  }
  ASSERT_EQ(held[0].u->refcount, 2);  // frame + test

  mf.clearAllImages();
  mf.clearAllDepthImages();

  for (size_t c = 0; c < 2; ++c) {
    EXPECT_TRUE(mf.image(c).empty());
    EXPECT_EQ(mf.image(c).data, nullptr);
    EXPECT_TRUE(mf.depthImage(c).empty());
    EXPECT_EQ(mf.numKeypoints(c), 3u);
    EXPECT_EQ(mf.descriptors(c).rows, 3);
    EXPECT_EQ(mf.keypointDescriptor(c, 2)[47], uchar(7 + c));
    EXPECT_EQ(mf.landmarkId(c, 1), 42u + c);
    EXPECT_EQ(mf.landmarkId(c, 0), 0u);
  }
  EXPECT_EQ(held[0].u->refcount, 1);  // the frame's reference is gone
  EXPECT_EQ(held[0].at<uchar>(0, 0), 100);
}
