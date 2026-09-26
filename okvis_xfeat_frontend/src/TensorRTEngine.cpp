/**
 * @file TensorRTEngine.cpp
 * @brief XFeat TensorRT runtime wrapper. Real impl under OKVIS_XFEAT_USE_TENSORRT
 *        (TensorRT 10.3 name-based I/O: getNbIOTensors / getIOTensorName /
 *        getTensorIOMode / getTensorShape / getTensorDataType, setTensorAddress
 *        + enqueueV3 — enqueueV2/bindings[] no longer exist), stub otherwise.
 *        First compiled against the device headers and run on the Orin Nano in
 *        T-0111.
 *
 * Expects the static engine exported by ../../xfeat_lightglue_onnx_mowe/export.py:
 *   input  "images"      [B,1,H,W] float (raw 0..255, NCHW mono), B = 2 (L/R)
 *   output "keypoints"   [B,K,2]   INT32 pixel coords (fp16-safe; a float coord
 *                                  output overflows the index-decode in fp16)
 *   output "descriptors" [B,K,64]  float
 *   output "scores"      [B,K]     float, -1 in padding slots
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

const char* dtype_name(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT: return "float32";
    case nvinfer1::DataType::kHALF: return "float16";
    case nvinfer1::DataType::kINT32: return "int32";
    case nvinfer1::DataType::kINT64: return "int64";
    default: return "other";
  }
}

std::string dims_str(const nvinfer1::Dims& d) {
  std::string s = "[";
  for (int i = 0; i < d.nbDims; ++i) s += (i ? "," : "") + std::to_string(d.d[i]);
  return s + "]";
}
}  // namespace

struct TensorRTEngine::Impl {
  Logger logger;
  nvinfer1::IRuntime* runtime = nullptr;
  nvinfer1::ICudaEngine* engine = nullptr;
  nvinfer1::IExecutionContext* context = nullptr;

  std::string input_name;
  std::uint32_t in_w = 0, in_h = 0, batch = 0, topk = 0;
  bool io_ok = false;
  std::vector<std::string> summary;

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

  // Introspect named I/O against the export.py contract, allocate output
  // buffers, cache input dims + batch + top-K.
  using nvinfer1::DataType;
  bool in_ok = false, kp_ok = false, sc_ok = false, de_ok = false;
  int n_io = 0;
  const int n = impl_->engine->getNbIOTensors();
  for (int i = 0; i < n; ++i) {
    const char* name = impl_->engine->getIOTensorName(i);
    const nvinfer1::Dims d = impl_->engine->getTensorShape(name);
    const DataType t = impl_->engine->getTensorDataType(name);
    const bool is_input = impl_->engine->getTensorIOMode(name) ==
                          nvinfer1::TensorIOMode::kINPUT;
    impl_->summary.push_back(std::string(is_input ? "in " : "out ") + name + ":" +
                             dtype_name(t) + ":" + dims_str(d));
    ++n_io;
    const std::string sname(name);
    if (is_input) {
      impl_->input_name = name;
      if (d.nbDims == 4) {
        impl_->batch = static_cast<std::uint32_t>(d.d[0]);
        impl_->in_h = static_cast<std::uint32_t>(d.d[2]);
        impl_->in_w = static_cast<std::uint32_t>(d.d[3]);
      }
      in_ok = sname == "images" && t == DataType::kFLOAT && d.nbDims == 4 && d.d[1] == 1;
      continue;
    }
    const std::size_t bytes = static_cast<std::size_t>(volume(d)) * 4;  // int32/float
    void* ptr = nullptr;
    if (cudaMalloc(&ptr, bytes) != cudaSuccess) {
      std::cerr << "[trt] cudaMalloc failed for output '" << name << "'\n";
      return false;
    }
    if (sname == "keypoints") {
      impl_->d_keypoints = ptr;
      if (d.nbDims == 3) impl_->topk = static_cast<std::uint32_t>(d.d[1]);  // [B,K,2]
      kp_ok = t == DataType::kINT32 && d.nbDims == 3 && d.d[2] == 2;
    } else if (sname == "scores") {
      impl_->d_scores = ptr;
      sc_ok = t == DataType::kFLOAT && d.nbDims == 2;
    } else if (sname == "descriptors") {
      impl_->d_descriptors = ptr;
      de_ok = t == DataType::kFLOAT && d.nbDims == 3 && d.d[2] == 64;
    } else {
      std::cerr << "[trt] unexpected output tensor '" << name << "'\n";
      cudaFree(ptr);
    }
  }
  impl_->io_ok = in_ok && kp_ok && sc_ok && de_ok && n_io == 4;
  for (const auto& s : impl_->summary) std::cerr << "[trt] " << s << "\n";
  if (!impl_->io_ok) std::cerr << "[trt] WARNING: I/O tensors do not match the export.py contract\n";

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
std::uint32_t TensorRTEngine::batch() const noexcept { return impl_->batch; }
std::uint32_t TensorRTEngine::topk() const noexcept { return impl_->topk; }
bool TensorRTEngine::io_names_match() const noexcept { return impl_->io_ok; }
const std::vector<std::string>& TensorRTEngine::tensor_summary() const noexcept {
  return impl_->summary;
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

  out.keypoints = static_cast<const std::int32_t*>(impl_->d_keypoints);
  out.scores = static_cast<const float*>(impl_->d_scores);
  out.descriptors = static_cast<const float*>(impl_->d_descriptors);
  out.count = impl_->topk;
  out.batch = impl_->batch;
  return true;  // outputs are on `stream`; caller syncs before readback
}

#else  // ---------- stub build (no TensorRT) ----------

struct TensorRTEngine::Impl {
  std::vector<std::string> summary;
};

TensorRTEngine::TensorRTEngine() : impl_(std::make_unique<Impl>()) {}
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
std::uint32_t TensorRTEngine::batch() const noexcept { return 0; }
std::uint32_t TensorRTEngine::topk() const noexcept { return 0; }
bool TensorRTEngine::io_names_match() const noexcept { return false; }
const std::vector<std::string>& TensorRTEngine::tensor_summary() const noexcept {
  return impl_->summary;
}
bool TensorRTEngine::infer(const void*, void*, EngineOutputs&) { return false; }

#endif  // OKVIS_XFEAT_USE_TENSORRT

}  // namespace xfeat
}  // namespace okvis
