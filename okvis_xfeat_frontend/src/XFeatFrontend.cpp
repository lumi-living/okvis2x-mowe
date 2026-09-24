/**
 * @file XFeatFrontend.cpp
 * @brief FrameBundle → CUDA preprocess → TensorRT XFeat → FrameFeatures.
 *
 * Real pipeline under OKVIS_XFEAT_USE_TENSORRT (verify on the Jetson); a stub
 * that emits empty features otherwise so the module builds on a dev box. Per
 * plane: resolve a CUDA pointer for the captured pixels (zero-copy NvBufSurface
 * map, else a host upload), run the preprocess kernel into the engine input,
 * enqueue, then read keypoints/scores/descriptors back and threshold on score.
 */
#include "okvis/xfeat/XFeatFrontend.hpp"

#include <iostream>

#include "mowe_camera/frame.hpp"
#include "mowe_camera/gpu_map.hpp"

#include "TensorRTEngine.hpp"

#ifdef OKVIS_XFEAT_USE_TENSORRT
#include <cuda_runtime.h>

#include "preprocess.h"
#endif

namespace okvis {
namespace xfeat {

struct XFeatFrontend::Impl {
  XFeatConfig cfg;
  TensorRTEngine engine;
  void* stream = nullptr;  // cudaStream_t

#ifdef OKVIS_XFEAT_USE_TENSORRT
  float* d_input = nullptr;       // engine input [1,1,H,W] float
  std::uint8_t* d_src_u8 = nullptr;  // host-upload staging (pitch == in_w)
  std::vector<std::int32_t> h_keypoints;  // int32 pixel coords (fp16-safe)
  std::vector<float> h_scores, h_descriptors;
#endif

  explicit Impl(const XFeatConfig& c) : cfg(c) {
#ifdef OKVIS_XFEAT_USE_TENSORRT
    cudaSetDevice(cfg.cuda_device);
    cudaStreamCreate(reinterpret_cast<cudaStream_t*>(&stream));
#endif
    if (!cfg.engine_path.empty()) engine.load(cfg.engine_path, cfg.cuda_device);
    if (cfg.input_width == 0 || cfg.input_height == 0) {
      engine.input_dims(cfg.input_width, cfg.input_height);
    }
#ifdef OKVIS_XFEAT_USE_TENSORRT
    if (engine.loaded()) allocate_buffers();
#endif
  }

  ~Impl() {
#ifdef OKVIS_XFEAT_USE_TENSORRT
    cudaFree(d_input);
    cudaFree(d_src_u8);
    if (stream) cudaStreamDestroy(static_cast<cudaStream_t>(stream));
#endif
  }

#ifdef OKVIS_XFEAT_USE_TENSORRT
  void allocate_buffers() {
    const std::size_t px = std::size_t(cfg.input_width) * cfg.input_height;
    cudaMalloc(&d_input, px * sizeof(float));
    cudaMalloc(&d_src_u8, px * sizeof(std::uint8_t));
  }

  // Read the engine outputs (on `stream`) back to host and fill `sf`, keeping
  // only keypoints whose (keypointness × reliability) score clears the
  // detection threshold. Padding slots carry score -1; low-texture heatmap
  // maxima carry a tiny softmax-floor score — both fall below the threshold.
  void readback(const EngineOutputs& o, StreamFeatures& sf) {
    const std::uint32_t K = o.count;
    h_keypoints.resize(K * 2);
    h_scores.resize(K);
    h_descriptors.resize(std::size_t(K) * kDescriptorDim);
    auto s = static_cast<cudaStream_t>(stream);
    cudaMemcpyAsync(h_keypoints.data(), o.keypoints,
                    K * 2 * sizeof(std::int32_t), cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(h_scores.data(), o.scores, K * sizeof(float),
                    cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(h_descriptors.data(), o.descriptors,
                    h_descriptors.size() * sizeof(float),
                    cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);

    sf.keypoints_px.reserve(K);
    sf.scores.reserve(K);
    sf.descriptors.reserve(h_descriptors.size());
    // Keypoints arrive as int32 pixel coords (fp16-safe export contract, see
    // ADR-0040); just widen to float for the POD boundary type.
    for (std::uint32_t i = 0; i < K; ++i) {
      if (h_scores[i] < cfg.score_threshold) continue;
      sf.keypoints_px.push_back(
          {static_cast<float>(h_keypoints[i * 2]),
           static_cast<float>(h_keypoints[i * 2 + 1])});
      sf.scores.push_back(h_scores[i]);
      const float* d = &h_descriptors[std::size_t(i) * kDescriptorDim];
      sf.descriptors.insert(sf.descriptors.end(), d, d + kDescriptorDim);
    }
  }

  // Resolve a CUDA device pointer (+ row pitch) for the plane's pixels.
  const void* resolve_device_input(const mowe::camera::ImagePlane& plane,
                                   mowe::camera::CudaImageMapping& map_token,
                                   int& pitch_bytes) {
    if (plane.gpu.has_gpu_backing() &&
        mowe::camera::map_plane_to_cuda(plane, map_token)) {
      pitch_bytes = static_cast<int>(map_token.pitch_bytes());
      return map_token.device_ptr();  // zero-copy
    }
    if (plane.data) {  // host upload fallback
      pitch_bytes = static_cast<int>(cfg.input_width);
      return upload_host(plane.data, plane.stride_bytes);
    }
    return nullptr;
  }

  // Stage host mono8 pixels into d_src_u8 (pitch == input_width).
  const void* upload_host(const std::uint8_t* data, std::uint32_t stride_bytes) {
    cudaMemcpy2DAsync(d_src_u8, cfg.input_width, data, stride_bytes,
                      cfg.input_width, cfg.input_height, cudaMemcpyHostToDevice,
                      static_cast<cudaStream_t>(stream));
    return d_src_u8;
  }

  // Preprocess `src` (device mono8, `pitch` bytes/row) + enqueue + readback.
  void run(const void* src, int pitch, StreamFeatures& sf) {
    xfeat_preprocess_mono_u8(src, pitch, d_input, cfg.input_width,
                             cfg.input_height, stream);
    EngineOutputs out;
    if (engine.infer(d_input, stream, out)) readback(out, sf);
  }
#endif  // OKVIS_XFEAT_USE_TENSORRT
};

XFeatFrontend::XFeatFrontend(const XFeatConfig& cfg)
    : impl_(std::make_unique<Impl>(cfg)) {
  if (!impl_->engine.loaded()) {
    std::cerr << "[xfeat] running in STUB mode (no engine) — features empty\n";
  }
}
XFeatFrontend::~XFeatFrontend() = default;
XFeatFrontend::XFeatFrontend(XFeatFrontend&&) noexcept = default;
XFeatFrontend& XFeatFrontend::operator=(XFeatFrontend&&) noexcept = default;

bool XFeatFrontend::engine_loaded() const noexcept {
  return impl_->engine.loaded();
}

void XFeatFrontend::input_dims(std::uint32_t& width,
                               std::uint32_t& height) const noexcept {
  width = impl_->cfg.input_width;
  height = impl_->cfg.input_height;
}

FrameFeatures XFeatFrontend::extract(const mowe::camera::FrameBundle& bundle) {
  FrameFeatures result;
  result.sequence = bundle.sequence();
  result.timestamp_ns = bundle.timestamp().count();

  for (const auto& plane : bundle.planes()) {
    StreamFeatures sf;
    sf.stream_id = plane.stream_id;

#ifdef OKVIS_XFEAT_USE_TENSORRT
    if (impl_->engine.loaded()) {
      // Static engine: the plane must match the engine's input dims.
      if (plane.width != impl_->cfg.input_width ||
          plane.height != impl_->cfg.input_height) {
        std::cerr << "[xfeat] plane '" << plane.stream_id << "' "
                  << plane.width << "x" << plane.height << " != engine "
                  << impl_->cfg.input_width << "x" << impl_->cfg.input_height
                  << " — skipped (re-export the engine for this size)\n";
        result.streams.push_back(std::move(sf));
        continue;
      }
      mowe::camera::CudaImageMapping map_token;
      int pitch = 0;
      const void* src = impl_->resolve_device_input(plane, map_token, pitch);
      if (src) {
        impl_->run(src, pitch, sf);
      }
    }
#endif  // OKVIS_XFEAT_USE_TENSORRT

    result.streams.push_back(std::move(sf));
  }
  return result;
}

StreamFeatures XFeatFrontend::extract_image(const std::uint8_t* data,
                                            std::uint32_t stride_bytes,
                                            std::uint32_t width,
                                            std::uint32_t height) {
  StreamFeatures sf;
#ifdef OKVIS_XFEAT_USE_TENSORRT
  if (!impl_->engine.loaded() || !data) return sf;
  if (width != impl_->cfg.input_width || height != impl_->cfg.input_height) {
    std::cerr << "[xfeat] image " << width << "x" << height << " != engine "
              << impl_->cfg.input_width << "x" << impl_->cfg.input_height
              << " — skipped (re-export the engine for this size)\n";
    return sf;
  }
  impl_->run(impl_->upload_host(data, stride_bytes),
             static_cast<int>(impl_->cfg.input_width), sf);
#else
  (void)data;
  (void)stride_bytes;
  (void)width;
  (void)height;
#endif  // OKVIS_XFEAT_USE_TENSORRT
  return sf;
}

}  // namespace xfeat
}  // namespace okvis
