/**
 * @file DescriptorDistance.hpp
 * @brief Descriptor distance seam: BRISK Hamming (48 B) or cosine on 64-D
 *        unit-norm float rows (256 B). Mow-e ADR-0040 / T-0112.
 *
 * A functor seam rather than a template over the estimator (mowe-nav-kb 01
 * §least-invasive): the Frontend matching loops keep working on raw
 * `const unsigned char*` descriptor rows and call `DescriptorMetric::operator()`.
 * Descriptors are L2-normalised, so ‖a−b‖² = 2(1 − a·b) and ranking by
 * `1 − a·b` (cosine distance, range [0, 2]) ranks identically to L2.
 */
#ifndef INCLUDE_OKVIS_DESCRIPTORDISTANCE_HPP_
#define INCLUDE_OKVIS_DESCRIPTORDISTANCE_HPP_

#include <cstddef>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <brisk/brisk.h>

namespace okvis {

/// BRISK descriptor row: 3 x 128 bit.
constexpr size_t kBriskDescriptorBytes = 48;
/// XFeat descriptor row: 64 x float32 (okvis::xfeat::kDescriptorDim).
constexpr int kFloatDescriptorDim = 64;
constexpr size_t kFloatDescriptorBytes = sizeof(float) * kFloatDescriptorDim;

/// Hamming distance between two 48-byte BRISK rows (bit count).
inline double hammingDistance(const unsigned char* a, const unsigned char* b) {
  return double(brisk::Hamming::PopcntofXORed(a, b, 3));  // 3 x 128 bit.
}

/// Cosine distance 1 − <a,b> between two unit-norm 64-D float rows, in [0, 2].
inline double cosineDistance(const unsigned char* a, const unsigned char* b) {
  const Eigen::Map<const Eigen::Matrix<float, kFloatDescriptorDim, 1>> fa(
      reinterpret_cast<const float*>(a));
  const Eigen::Map<const Eigen::Matrix<float, kFloatDescriptorDim, 1>> fb(
      reinterpret_cast<const float*>(b));
  return 1.0 - double(fa.dot(fb));
}

/// \brief Runtime-selected descriptor metric.
struct DescriptorMetric {
  bool isFloat = false;  ///< false: BRISK/Hamming, true: XFeat float/cosine.

  /// Bytes per descriptor row (== cv::Mat::step of the descriptor matrix).
  size_t bytes() const { return isFloat ? kFloatDescriptorBytes : kBriskDescriptorBytes; }

  /// Distance between two rows; the matching_threshold is on this scale
  /// (Hamming bits, or cosine distance).
  double operator()(const unsigned char* a, const unsigned char* b) const {
    return isFloat ? cosineDistance(a, b) : hammingDistance(a, b);
  }

  /// Metric implied by a descriptor matrix (CV_32F rows → float/cosine).
  static DescriptorMetric forDescriptors(const cv::Mat& descriptors) {
    return DescriptorMetric{descriptors.depth() == CV_32F};
  }
};

}  // namespace okvis

#endif  // INCLUDE_OKVIS_DESCRIPTORDISTANCE_HPP_
