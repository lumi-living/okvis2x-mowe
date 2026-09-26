/**
 * @file TensorRTEngine.hpp
 * @brief Thin TensorRT 10 runtime wrapper for the static XFeat engine.
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
#include <vector>

namespace okvis {
namespace xfeat {

/// Device-pointer views into the engine's output bindings after an inference.
/// Slot b of the batch starts at keypoints + b*K*2, scores + b*K,
/// descriptors + b*K*64.
struct EngineOutputs {
  const std::int32_t* keypoints = nullptr;  ///< [B x K x 2] int32 (x,y) engine px
  const float* scores = nullptr;            ///< [B x K]     -1 in padding slots
  const float* descriptors = nullptr;       ///< [B x K x 64]
  std::uint32_t count = 0;                  ///< K (static top-K)
  std::uint32_t batch = 0;                  ///< B
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

  /// Static input dims / batch / top-K the engine was built for (0 in stub).
  void input_dims(std::uint32_t& width, std::uint32_t& height) const noexcept;
  std::uint32_t batch() const noexcept;
  std::uint32_t topk() const noexcept;

  /// True when the engine's I/O tensors are exactly the export.py contract:
  /// input "images" float [B,1,H,W]; outputs "keypoints" int32 [B,K,2],
  /// "descriptors" float [B,K,64], "scores" float [B,K]. // T-0111
  bool io_names_match() const noexcept;
  /// "name:dtype:dims" per I/O tensor, in engine order (diagnostics/JSON).
  const std::vector<std::string>& tensor_summary() const noexcept;

  /// Enqueue one inference. `input_device_ptr` is the preprocessed NCHW float
  /// input [B,1,H,W] on the device; `stream` is a cudaStream_t (void* to keep
  /// CUDA out of this header). Outputs are on `stream`; the caller syncs
  /// before reading them back. Returns false in stub builds.
  bool infer(const void* input_device_ptr, void* stream, EngineOutputs& out);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace xfeat
}  // namespace okvis

#endif  // OKVIS_XFEAT_TENSORRTENGINE_HPP_
