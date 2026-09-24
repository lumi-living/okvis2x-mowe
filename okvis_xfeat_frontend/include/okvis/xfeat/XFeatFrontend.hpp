/**
 * @file okvis/xfeat/XFeatFrontend.hpp
 * @brief XFeat-on-TensorRT feature frontend for OKVIS.
 *
 * Pipeline (SKELETON — see README for what is wired vs. stubbed):
 *
 *   mowe::camera::FrameBundle           (from mowe_camera_core)
 *        │  per plane
 *        ▼
 *   map DMABUF/NvBufSurface → CUDA      (mowe_camera/gpu_map.hpp, zero-copy)
 *        ▼
 *   CUDA preprocess (cast/normalise/pad → engine input binding)   [TODO kernel]
 *        ▼
 *   TensorRT XFeat engine enqueue       (TensorRTEngine)
 *        ▼
 *   okvis::xfeat::FrameFeatures         → OKVIS adapter (next step)
 *
 * This replaces BRISK detect+describe. It does NOT use OKVIS's USE_NN path —
 * that is a Torch semantic-segmentation (sky/person) mask, an orthogonal
 * concern. XFeat is a genuinely separate frontend. The LighterGlue matcher and
 * the OKVIS MultiFrame hand-off are separate workstreams (see README).
 */
#ifndef OKVIS_XFEAT_XFEATFRONTEND_HPP_
#define OKVIS_XFEAT_XFEATFRONTEND_HPP_

#include <cstdint>
#include <memory>
#include <string>

#include "okvis/xfeat/XFeatFeatures.hpp"

namespace mowe {
namespace camera {
class FrameBundle;  // mowe_camera/frame.hpp
}  // namespace camera
}  // namespace mowe

namespace okvis {
namespace xfeat {

struct XFeatConfig {
  /// Path to the serialized TensorRT engine (.plan) built from the XFeat ONNX
  /// (see ../xfeat_lightglue_onnx + trtexec). Empty → run in stub mode.
  std::string engine_path;

  /// The static input dims the engine was built for (TensorRT prefers static
  /// shapes; the camera frames are letterboxed/padded to this). 0 → infer from
  /// the engine bindings at load.
  std::uint32_t input_width = 0;
  std::uint32_t input_height = 0;

  /// Fixed top-K the engine emits (static-shape export). Must match the ONNX.
  std::uint32_t max_keypoints = 4096;

  /// Detection-score floor: keep a keypoint only when (keypointness ×
  /// reliability) exceeds this. The static export pads its fixed top-K with
  /// low-texture heatmap maxima (tiny but >0 softmax floor) — without a real
  /// threshold those survive as a flickering noise band over blank walls/sky.
  /// 0.05 matches XFeat's upstream `detection_threshold`. // ADR-0040.
  float score_threshold = 0.05f;

  /// Run on this CUDA device.
  int cuda_device = 0;
};

/// \brief Owns the TensorRT engine + CUDA stream; extracts XFeat features from
/// camera frames. Not copyable (owns GPU resources); movable.
class XFeatFrontend {
 public:
  explicit XFeatFrontend(const XFeatConfig& cfg);
  ~XFeatFrontend();
  XFeatFrontend(XFeatFrontend&&) noexcept;
  XFeatFrontend& operator=(XFeatFrontend&&) noexcept;
  XFeatFrontend(const XFeatFrontend&) = delete;
  XFeatFrontend& operator=(const XFeatFrontend&) = delete;

  /// True when a real TensorRT engine is loaded (vs. stub mode).
  bool engine_loaded() const noexcept;

  /// The static input dims of the loaded engine (0/0 in stub mode).
  void input_dims(std::uint32_t& width, std::uint32_t& height) const noexcept;

  /// Run XFeat on every plane of `bundle`. On the Jetson zero-copy path each
  /// plane's NvBufSurface is mapped straight to CUDA (no host round-trip); on a
  /// host-only build it falls back to an upload (TODO). Thread-compatible: call
  /// from a single inference thread (one engine context, one stream).
  FrameFeatures extract(const mowe::camera::FrameBundle& bundle);

  /// Run XFeat on a single host-resident mono8 image (the ROS-subscriber /
  /// cv::Mat path — no FrameBundle involved). `data` points at `height` rows of
  /// `width` pixels, `stride_bytes` apart. Dims must equal the engine's static
  /// input dims. Same threading contract as extract().
  StreamFeatures extract_image(const std::uint8_t* data,
                               std::uint32_t stride_bytes,
                               std::uint32_t width, std::uint32_t height);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace xfeat
}  // namespace okvis

#endif  // OKVIS_XFEAT_XFEATFRONTEND_HPP_
