/**
 * @file OrbFeatures.hpp
 * @brief cv::ORB behind the BRISK detector/extractor seam (Mow-e T-0134).
 *
 * frontend_parameters.detector: orb swaps BRISK detect+describe for OpenCV's
 * ORB (FAST on an 8-level 1.2 pyramid + rotated BRIEF, 256 bit). The 32-byte
 * ORB rows are zero-padded to the 48-byte BRISK row, so the whole binary path
 * (storage, Hamming matching over 3 x 128 bit, loop-closure verification)
 * is untouched: XOR of the zero pads is 0, the distance is ORB's own 256-bit
 * Hamming distance. The float/XFeat path is not involved.
 */
#ifndef INCLUDE_OKVIS_ORBFEATURES_HPP_
#define INCLUDE_OKVIS_ORBFEATURES_HPP_

#include <cmath>
#include <vector>

#include <opencv2/features2d/features2d.hpp>

#include <okvis/DescriptorDistance.hpp>

namespace okvis {

/// ORB descriptor bytes (256 bit) inside the 48-byte BRISK row.
constexpr int kOrbDescriptorBytes = 32;

/// One cv::Feature2D used both as detector and extractor (a separate instance
/// per role and camera, like BRISK, so no instance is shared across threads).
class OrbFeature2D : public cv::Feature2D {
 public:
  static constexpr float kScaleFactor = 1.2f;
  static constexpr int kLevels = 8;
  static constexpr int kPatchSize = 31;

  /// @param maxKeypoints  ORB nfeatures (= max_num_keypoints).
  /// @param fastThreshold ORB fastThreshold (= detection_threshold).
  OrbFeature2D(int maxKeypoints, int fastThreshold)
      : orb_(cv::ORB::create(maxKeypoints, kScaleFactor, kLevels, kPatchSize, 0, 2,
                             cv::ORB::HARRIS_SCORE, kPatchSize, fastThreshold)) {}

  void detectAndCompute(cv::InputArray image, cv::InputArray mask,
                        std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors,
                        bool useProvidedKeypoints = false) override {
    if (useProvidedKeypoints) {
      for (auto& kp : keypoints) kp.size = kPatchSize * scale(kp.octave);  // ORB's own convention
    }
    cv::Mat d;
    if (descriptors.needed()) {
      orb_->detectAndCompute(image, mask, keypoints, d, useProvidedKeypoints);
    } else {
      orb_->detectAndCompute(image, mask, keypoints, cv::noArray(), useProvidedKeypoints);
    }
    // Keypoint size from the octave on BRISK's scale: the adapters use
    // sigma = 0.8 * size / 12 px, i.e. 0.8 px at full resolution (T-0134).
    for (auto& kp : keypoints) kp.size = 12.f * scale(kp.octave);
    if (descriptors.needed()) {
      cv::Mat padded = cv::Mat::zeros(d.rows, int(kBriskDescriptorBytes), CV_8UC1);
      if (d.rows > 0) d.copyTo(padded.colRange(0, kOrbDescriptorBytes));
      padded.copyTo(descriptors);
    }
  }

  int descriptorSize() const override { return int(kBriskDescriptorBytes); }
  int descriptorType() const override { return CV_8UC1; }
  int defaultNorm() const override { return cv::NORM_HAMMING; }

 private:
  static float scale(int octave) { return std::pow(kScaleFactor, float(octave)); }
  cv::Ptr<cv::ORB> orb_;
};

}  // namespace okvis

#endif  // INCLUDE_OKVIS_ORBFEATURES_HPP_
