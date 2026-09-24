/**
 * @file TensorRTEngine.cpp
 * @brief XFeat TensorRT runtime wrapper. Real impl under OKVIS_XFEAT_USE_TENSORRT
 *        (TensorRT 10.3 API), stub otherwise.
 *
 * Verified against the TensorRT 10.x name-based I/O API: getNbIOTensors /
 * getIOTensorName / getTensorIOMode / getTensorShape, setTensorAddress +
 * enqueueV3 (enqueueV2/bindings[] are removed in TRT 10). // confirm on device.
 *
 * Expects the static engine exported by ../../xfeat_lightglue_onnx_mowe/export.py:
 *   input  "images"      [1,1,H,W] float (raw 0..255, NCHW mono)
 *   output "keypoints"   [1,K,2]   INT32 pixel coords (fp16-safe; a float coord
 *                                  output overflows the index-decode in fp16)
 *   output "descriptors" [1,K,64]  float
 *   output "scores"      [1,K]     float
 */
#include "TensorRTEngine.hpp"

#include <iostream>

namespace okvis {
namespace xfeat {

#ifdef OKVIS_XFEAT_USE_TENSORRT

}  // namespace xfeat
}  // namespace okvis

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <fstream>
#include <string>
#include <vector>

namespace okvis {
namespace xfeat {

namespace {
class Logger : public nvinfer1::ILogger {
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kWARNING) std::cerr << "[trt] " << msg << "\n";
  }
};

int64_t volume(const nvinfer1::Dims& d) {
  int64_t v = 1;
  for (int i = 0; i < d.nbDims; ++i) v *= d.d[i];
  return v;
}
}  // namespace

struct TensorRTEngine::Impl {
  Logger logger;
  nvinfer1::IRuntime* runtime = nullptr;
  nvinfer1::ICudaEngine* engine = nullptr;
  nvinfer1::IExecutionContext* context = nullptr;

  std::string input_name;
  std::uint32_t in_w = 0, in_h = 0;
  std::uint32_t topk = 0;

  // Output device buffers, allocated to their static volumes at load.
  void* d_keypoints = nullptr;
  void* d_scores = nullptr;
  void* d_descriptors = nullptr;

  ~Impl() {
    cudaFree(d_keypoints);
    cudaFree(d_scores);
    cudaFree(d_descriptors);
    delete context;  // TRT 10: objects are deleted, not destroy()'d.
    delete engine;
    delete runtime;
  }
};

TensorRTEngine::TensorRTEngine() : impl_(std::make_unique<Impl>()) {}
TensorRTEngine::~TensorRTEngine() = default;
TensorRTEngine::TensorRTEngine(TensorRTEngine&&) noexcept = default;
TensorRTEngine& TensorRTEngine::operator=(TensorRTEngine&&) noexcept = default;

bool TensorRTEngine::load(const std::string& engine_path, int cuda_device) {
  cudaSetDevice(cuda_device);

  std::ifstream f(engine_path, std::ios::binary);
  if (!f) {
    std::cerr << "[trt] cannot open engine: " << engine_path << "\n";
    return false;
  }
  std::vector<char> blob((std::istreambuf_iterator<char>(f)), {});

  impl_->runtime = nvinfer1::createInferRuntime(impl_->logger);
  if (!impl_->runtime) return false;
  impl_->engine =
      impl_->runtime->deserializeCudaEngine(blob.data(), blob.size());
  if (!impl_->engine) return false;
  impl_->context = impl_->engine->createExecutionContext();
  if (!impl_->context) return false;

  // Introspect named I/O, allocate output buffers, cache input dims + top-K.
  const int n = impl_->engine->getNbIOTensors();
  for (int i = 0; i < n; ++i) {
    const char* name = impl_->engine->getIOTensorName(i);
    const nvinfer1::Dims d = impl_->engine->getTensorShape(name);
    const bool is_input = impl_->engine->getTensorIOMode(name) ==
                          nvinfer1::TensorIOMode::kINPUT;
    if (is_input) {
      impl_->input_name = name;
      impl_->in_h = static_cast<std::uint32_t>(d.d[d.nbDims - 2]);
      impl_->in_w = static_cast<std::uint32_t>(d.d[d.nbDims - 1]);
      continue;
    }
    const std::size_t bytes = static_cast<std::size_t>(volume(d)) * sizeof(float);
    void* ptr = nullptr;
    if (cudaMalloc(&ptr, bytes) != cudaSuccess) {
      std::cerr << "[trt] cudaMalloc failed for output '" << name << "'\n";
      return false;
    }
    const std::string sname(name);
    if (sname == "keypoints") {
      impl_->d_keypoints = ptr;
      impl_->topk = static_cast<std::uint32_t>(d.d[1]);  // [1,K,2]
    } else if (sname == "scores") {
      impl_->d_scores = ptr;
    } else if (sname == "descriptors") {
      impl_->d_descriptors = ptr;
    } else {
      std::cerr << "[trt] unexpected output tensor '" << name << "'\n";
      cudaFree(ptr);
    }
  }

  if (!impl_->d_keypoints || !impl_->d_scores || !impl_->d_descriptors) {
    std::cerr << "[trt] engine missing keypoints/scores/descriptors outputs\n";
    return false;
  }
  return true;
}

bool TensorRTEngine::loaded() const noexcept { return impl_->engine != nullptr; }

void TensorRTEngine::input_dims(std::uint32_t& w, std::uint32_t& h) const noexcept {
  w = impl_->in_w;
  h = impl_->in_h;
}

bool TensorRTEngine::infer(const void* input_device_ptr, void* stream,
                           EngineOutputs& out) {
  if (!impl_->context || !input_device_ptr) return false;

  impl_->context->setTensorAddress(impl_->input_name.c_str(),
                                   const_cast<void*>(input_device_ptr));
  impl_->context->setTensorAddress("keypoints", impl_->d_keypoints);
  impl_->context->setTensorAddress("scores", impl_->d_scores);
  impl_->context->setTensorAddress("descriptors", impl_->d_descriptors);

  if (!impl_->context->enqueueV3(static_cast<cudaStream_t>(stream))) {
    return false;
  }

  out.keypoints = static_cast<const float*>(impl_->d_keypoints);
  out.scores = static_cast<const float*>(impl_->d_scores);
  out.descriptors = static_cast<const float*>(impl_->d_descriptors);
  out.count = impl_->topk;
  return true;  // outputs are on `stream`; caller syncs before readback
}

#else  // ---------- stub build (no TensorRT) ----------

struct TensorRTEngine::Impl {};

TensorRTEngine::TensorRTEngine() : impl_(nullptr) {}
TensorRTEngine::~TensorRTEngine() = default;
TensorRTEngine::TensorRTEngine(TensorRTEngine&&) noexcept = default;
TensorRTEngine& TensorRTEngine::operator=(TensorRTEngine&&) noexcept = default;

bool TensorRTEngine::load(const std::string&, int) {
  std::cerr << "[trt] built without OKVIS_XFEAT_USE_TENSORRT — stub engine\n";
  return false;
}
bool TensorRTEngine::loaded() const noexcept { return false; }
void TensorRTEngine::input_dims(std::uint32_t& w, std::uint32_t& h) const noexcept {
  w = 0;
  h = 0;
}
bool TensorRTEngine::infer(const void*, void*, EngineOutputs&) { return false; }

#endif  // OKVIS_XFEAT_USE_TENSORRT

}  // namespace xfeat
}  // namespace okvis
