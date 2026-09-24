/**
 * @file TensorRTEngine.hpp
 * @brief Thin TensorRT runtime wrapper for the XFeat engine (skeleton).
 *
 * Compiled with the real nvinfer/CUDA calls when OKVIS_XFEAT_USE_TENSORRT is
 * defined; otherwise a stub so the module builds on a non-Jetson dev box. All
 * GPU pointers are CUDA device pointers — input is produced by the preprocess
 * kernel, outputs are read back by the frontend.
 */
#ifndef OKVIS_XFEAT_TENSORRTENGINE_HPP_
#define OKVIS_XFEAT_TENSORRTENGINE_HPP_

#include <cstdint>
#include <memory>
#include <string>

namespace okvis {
namespace xfeat {

/// Device-pointer views into the engine's output bindings after an inference.
struct EngineOutputs {
  const float* keypoints = nullptr;    ///< [K x 2] device ptr (u,v)
  const float* scores = nullptr;       ///< [K]     device ptr
  const float* descriptors = nullptr;  ///< [K x 64] device ptr
  std::uint32_t count = 0;             ///< K (static top-K)
};

class TensorRTEngine {
 public:
  TensorRTEngine();
  ~TensorRTEngine();
  TensorRTEngine(TensorRTEngine&&) noexcept;
  TensorRTEngine& operator=(TensorRTEngine&&) noexcept;
  TensorRTEngine(const TensorRTEngine&) = delete;
  TensorRTEngine& operator=(const TensorRTEngine&) = delete;

  /// Deserialize a .plan. Returns false in stub builds or on failure.
  bool load(const std::string& engine_path, int cuda_device);

  bool loaded() const noexcept;

  /// Query the input binding dims the engine was built for (static shape).
  void input_dims(std::uint32_t& width, std::uint32_t& height) const noexcept;

  /// Enqueue one inference. `input_device_ptr` is the preprocessed NCHW float
  /// input on the device; `stream` is a cudaStream_t (void* to keep CUDA out of
  /// this header). Output device pointers are returned via `out`. Returns false
  /// in stub builds.
  bool infer(const void* input_device_ptr, void* stream, EngineOutputs& out);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace xfeat
}  // namespace okvis

#endif  // OKVIS_XFEAT_TENSORRTENGINE_HPP_
