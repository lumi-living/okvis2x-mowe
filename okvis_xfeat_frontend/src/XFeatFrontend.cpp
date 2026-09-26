/**
 * @file XFeatFrontend.cpp
 * @brief FrameBundle / host images → CUDA resize+preprocess → TensorRT XFeat
 *        (batch=2) → FrameFeatures.
 *
 * Real pipeline under OKVIS_XFEAT_USE_TENSORRT (first run on the Orin Nano in
 * T-0111); a stub that emits empty features otherwise so the module builds on
 * a dev box. Per eye: resolve a CUDA pointer for the pixels (zero-copy
 * NvBufSurface map, else an H2D copy from the host buffer — pinned when the
 * caller registered it), resize into that eye's batch slot of the engine
 * input, then one enqueueV3 for the pair, D2H readback, padding strip and
 * scale-back to full-res pixels (FrameFeaturesUtil.hpp).
 */
#include "okvis/xfeat/XFeatFrontend.hpp"

#include <iostream>

#include "mowe_camera/frame.hpp"
#include "mowe_camera/gpu_map.hpp"

#include "TensorRTEngine.hpp"
#include "okvis/xfeat/FrameFeaturesUtil.hpp"

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
  float* d_input = nullptr;          // engine input [B,1,H,W] float
  std::uint8_t* d_src_u8 = nullptr;  // host-upload staging, B slots of src_w*src_h
  std::uint32_t src_w = 0, src_h = 0;  // staging dims (reallocated on change)
  std::vector<std::int32_t> h_keypoints;
  std::vector<float> h_scores, h_descriptors;
  struct Slot {  // what went into each batch slot this round
    bool used = false;
    std::uint32_t w = 0, h = 0;  // source dims (for the scale-back)
  };
  std::vector<Slot> slots;
#endif

  explicit Impl(const XFeatConfig& c) : cfg(c) {
#ifdef OKVIS_XFEAT_USE_TENSORRT
    cudaSetDevice(cfg.cuda_device);
    // High-priority stream so perception engines cannot starve VIO tracking
    // (KB 03 §streams). greatest = numerically lowest.
    int lo = 0, hi = 0;
    cudaDeviceGetStreamPriorityRange(&lo, &hi);
    cudaStreamCreateWithPriority(reinterpret_cast<cudaStream_t*>(&stream),
                                 cudaStreamNonBlocking, hi);
#endif
    if (!cfg.engine_path.empty()) engine.load(cfg.engine_path, cfg.cuda_device);
    if (cfg.input_width == 0 || cfg.input_height == 0) {
      engine.input_dims(cfg.input_width, cfg.input_height);
    }
#ifdef OKVIS_XFEAT_USE_TENSORRT
    if (engine.loaded()) {
      cfg.max_keypoints = engine.topk();
      slots.resize(engine.batch());
      const std::size_t px = std::size_t(cfg.input_width) * cfg.input_height;
      cudaMalloc(&d_input, px * engine.batch() * sizeof(float));
    }
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
  cudaStream_t s() const { return static_cast<cudaStream_t>(stream); }
  std::size_t eng_px() const { return std::size_t(cfg.input_width) * cfg.input_height; }

  // Staging for host uploads: batch() slots of src_w x src_h (pitch == src_w).
  void ensure_staging(std::uint32_t w, std::uint32_t h) {
    if (w == src_w && h == src_h && d_src_u8) return;
    cudaFree(d_src_u8);
    src_w = w;
    src_h = h;
    cudaMalloc(&d_src_u8, std::size_t(w) * h * slots.size());
  }

  // Copy host mono8 pixels into staging slot `b`; returns the device pointer.
  const std::uint8_t* upload_host(std::uint32_t b, const std::uint8_t* data,
                                  std::uint32_t stride, std::uint32_t w,
                                  std::uint32_t h) {
    ensure_staging(w, h);
    std::uint8_t* dst = d_src_u8 + std::size_t(b) * w * h;
    cudaMemcpy2DAsync(dst, w, data, stride, w, h, cudaMemcpyHostToDevice, s());
    return dst;
  }

  // Resize `src` (device mono8, `pitch` bytes/row, w x h) into batch slot `b`.
  void preprocess(std::uint32_t b, const void* src, int pitch, std::uint32_t w,
                  std::uint32_t h) {
    xfeat_preprocess_resize_u8(src, pitch, int(w), int(h),
                               d_input + b * eng_px(), int(cfg.input_width),
                               int(cfg.input_height), stream);
    slots[b] = {true, w, h};
  }

  // One enqueue for all filled slots, D2H, then assemble each used slot.
  // `out[b]` receives slot b (must have slots.size() entries).
  bool run(StreamFeatures* out) {
    EngineOutputs o;
    if (!engine.infer(d_input, stream, o)) return false;
    const std::uint32_t K = o.count, B = o.batch;
    h_keypoints.resize(std::size_t(B) * K * 2);
    h_scores.resize(std::size_t(B) * K);
    h_descriptors.resize(std::size_t(B) * K * kDescriptorDim);
    cudaMemcpyAsync(h_keypoints.data(), o.keypoints,
                    h_keypoints.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost, s());
    cudaMemcpyAsync(h_scores.data(), o.scores, h_scores.size() * sizeof(float),
                    cudaMemcpyDeviceToHost, s());
    cudaMemcpyAsync(h_descriptors.data(), o.descriptors,
                    h_descriptors.size() * sizeof(float), cudaMemcpyDeviceToHost, s());
    if (cudaStreamSynchronize(s()) != cudaSuccess) {
      std::cerr << "[xfeat] CUDA error: " << cudaGetErrorString(cudaGetLastError()) << "\n";
      return false;
    }
    for (std::uint32_t b = 0; b < B; ++b) {
      if (!slots[b].used) continue;
      const ResizeScale scale = resize_scale(slots[b].w, slots[b].h,
                                             cfg.input_width, cfg.input_height);
      assemble_stream(&h_keypoints[std::size_t(b) * K * 2], &h_scores[std::size_t(b) * K],
                      &h_descriptors[std::size_t(b) * K * kDescriptorDim], K, scale,
                      cfg.score_threshold, out[b]);
      slots[b].used = false;
    }
    return true;
  }

  // Resolve a CUDA device pointer (+ row pitch) for a captured plane.
  const void* resolve_device_input(std::uint32_t b,
                                   const mowe::camera::ImagePlane& plane,
                                   mowe::camera::CudaImageMapping& map_token,
                                   int& pitch_bytes) {
    if (plane.gpu.has_gpu_backing() &&
        mowe::camera::map_plane_to_cuda(plane, map_token)) {
      pitch_bytes = static_cast<int>(map_token.pitch_bytes());
      return map_token.device_ptr();  // zero-copy (compiled, unused: T-0111)
    }
    if (plane.data) {  // host upload (pinned USERPTR path, KB 04 §3.1)
      pitch_bytes = static_cast<int>(plane.width);
      return upload_host(b, plane.data, plane.stride_bytes, plane.width, plane.height);
    }
    return nullptr;
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

bool XFeatFrontend::engine_loaded() const noexcept { return impl_->engine.loaded(); }

void XFeatFrontend::input_dims(std::uint32_t& width, std::uint32_t& height) const noexcept {
  width = impl_->cfg.input_width;
  height = impl_->cfg.input_height;
}
std::uint32_t XFeatFrontend::batch() const noexcept { return impl_->engine.batch(); }
std::uint32_t XFeatFrontend::max_keypoints() const noexcept { return impl_->engine.topk(); }
bool XFeatFrontend::io_names_match() const noexcept { return impl_->engine.io_names_match(); }
std::vector<std::string> XFeatFrontend::tensor_summary() const {
  return impl_->engine.tensor_summary();
}

bool XFeatFrontend::register_host_buffer(const void* ptr, std::size_t bytes) {
#ifdef OKVIS_XFEAT_USE_TENSORRT
  return cudaHostRegister(const_cast<void*>(ptr), bytes, cudaHostRegisterDefault) == cudaSuccess;
#else
  (void)ptr; (void)bytes;
  return false;
#endif
}
void XFeatFrontend::unregister_host_buffer(const void* ptr) {
#ifdef OKVIS_XFEAT_USE_TENSORRT
  cudaHostUnregister(const_cast<void*>(ptr));
#else
  (void)ptr;
#endif
}

FrameFeatures XFeatFrontend::extract(const mowe::camera::FrameBundle& bundle) {
  FrameFeatures result;
  result.sequence = bundle.sequence();
  result.timestamp_ns = bundle.timestamp().count();
  const auto planes = bundle.planes();
  for (const auto& plane : planes) {
    StreamFeatures sf;
    sf.stream_id = plane.stream_id;
    result.streams.push_back(std::move(sf));
  }

#ifdef OKVIS_XFEAT_USE_TENSORRT
  if (!impl_->engine.loaded()) return result;
  // Planes fill batch slots in order; more planes than slots → extra rounds.
  const std::uint32_t B = impl_->engine.batch();
  std::vector<mowe::camera::CudaImageMapping> tokens(B);
  for (std::size_t first = 0; first < planes.size(); first += B) {
    std::uint32_t filled = 0;
    for (std::uint32_t b = 0; b < B && first + b < planes.size(); ++b) {
      int pitch = 0;
      const void* src = impl_->resolve_device_input(b, planes[first + b], tokens[b], pitch);
      if (!src) continue;
      impl_->preprocess(b, src, pitch, planes[first + b].width, planes[first + b].height);
      ++filled;
    }
    if (filled) impl_->run(&result.streams[first]);
  }
#endif  // OKVIS_XFEAT_USE_TENSORRT
  return result;
}

StreamFeatures XFeatFrontend::extract_image(const std::uint8_t* data,
                                            std::uint32_t stride_bytes,
                                            std::uint32_t width,
                                            std::uint32_t height) {
  StreamFeatures sf;
#ifdef OKVIS_XFEAT_USE_TENSORRT
  if (!impl_->engine.loaded() || !data) return sf;
  std::vector<StreamFeatures> out(impl_->engine.batch());
  const std::uint8_t* src = impl_->upload_host(0, data, stride_bytes, width, height);
  impl_->preprocess(0, src, int(width), width, height);
  if (impl_->run(out.data())) sf = std::move(out[0]);
#else
  (void)data; (void)stride_bytes; (void)width; (void)height;
#endif  // OKVIS_XFEAT_USE_TENSORRT
  return sf;
}

FrameFeatures XFeatFrontend::extract_pair(const std::uint8_t* left, std::uint32_t left_stride,
                                          const std::uint8_t* right, std::uint32_t right_stride,
                                          std::uint32_t width, std::uint32_t height) {
  FrameFeatures result;
  result.streams.resize(2);
  result.streams[0].stream_id = "left";
  result.streams[1].stream_id = "right";
#ifdef OKVIS_XFEAT_USE_TENSORRT
  if (!impl_->engine.loaded() || !left || !right) return result;
  if (impl_->engine.batch() < 2) {
    std::cerr << "[xfeat] extract_pair needs a batch>=2 engine (have "
              << impl_->engine.batch() << ")\n";
    return result;
  }
  std::vector<StreamFeatures> out(impl_->engine.batch());
  impl_->preprocess(0, impl_->upload_host(0, left, left_stride, width, height), int(width), width, height);
  impl_->preprocess(1, impl_->upload_host(1, right, right_stride, width, height), int(width), width, height);
  if (impl_->run(out.data())) {
    out[0].stream_id = "left";
    out[1].stream_id = "right";
    result.streams[0] = std::move(out[0]);
    result.streams[1] = std::move(out[1]);
  }
#else
  (void)left; (void)left_stride; (void)right; (void)right_stride; (void)width; (void)height;
#endif
  return result;
}

}  // namespace xfeat
}  // namespace okvis
