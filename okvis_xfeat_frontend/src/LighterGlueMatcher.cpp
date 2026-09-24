/**
 * @file LighterGlueMatcher.cpp
 * @brief LighterGlue TensorRT pair matcher. Real impl under
 *        OKVIS_XFEAT_USE_TENSORRT (TensorRT 10.3), stub otherwise.
 *
 * The engine's "matches"/"scores" outputs have data-dependent shapes (the
 * in-graph mutual-NN keeps a variable number of pairs), so the execution
 * context needs an nvinfer1::IOutputAllocator per output. M <= K always, so
 * both outputs are preallocated at capacity and never grow at runtime;
 * notifyShape() delivers the actual M after enqueue.
 */
#include "okvis/xfeat/LighterGlueMatcher.hpp"

#include <iostream>

#ifdef OKVIS_XFEAT_USE_TENSORRT

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <fstream>

namespace okvis {
namespace xfeat {

namespace {

constexpr int kDescDim = 64;  // XFeat descriptor dimensionality.

class Logger : public nvinfer1::ILogger {
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kWARNING) std::cerr << "[trt-lg] " << msg << "\n";
  }
};

/// Preallocated-at-capacity output allocator for one data-dependent tensor.
class FixedCapacityOutput : public nvinfer1::IOutputAllocator {
 public:
  bool init(std::size_t capacity_bytes) {
    capacity_ = capacity_bytes;
    return cudaMalloc(&ptr_, capacity_bytes) == cudaSuccess;
  }
  ~FixedCapacityOutput() override { cudaFree(ptr_); }

  void* reallocateOutput(char const* /*name*/, void* /*current*/,
                         std::uint64_t size,
                         std::uint64_t /*alignment*/) noexcept override {
    // M <= K by construction (mutual-NN), so the preallocation always fits;
    // returning nullptr on overflow makes TRT fail the enqueue loudly rather
    // than write out of bounds.
    return size <= capacity_ ? ptr_ : nullptr;
  }
  void* reallocateOutputAsync(char const* name, void* current,
                              std::uint64_t size, std::uint64_t alignment,
                              cudaStream_t /*stream*/) noexcept override {
    return reallocateOutput(name, current, size, alignment);
  }
  void notifyShape(char const* /*name*/,
                   nvinfer1::Dims const& dims) noexcept override {
    shape_ = dims;
  }

  const void* data() const noexcept { return ptr_; }
  const nvinfer1::Dims& shape() const noexcept { return shape_; }

 private:
  void* ptr_ = nullptr;
  std::size_t capacity_ = 0;
  nvinfer1::Dims shape_{};
};

}  // namespace

struct LighterGlueMatcher::Impl {
  LighterGlueConfig cfg;
  Logger logger;
  nvinfer1::IRuntime* runtime = nullptr;
  nvinfer1::ICudaEngine* engine = nullptr;
  nvinfer1::IExecutionContext* context = nullptr;
  cudaStream_t stream = nullptr;

  std::uint32_t capacity = 0;  // K from the engine bindings.

  // Device input slates [1,K,2] / [1,K,64], zero-padded past n.
  float* d_kpts[2] = {nullptr, nullptr};
  float* d_desc[2] = {nullptr, nullptr};

  FixedCapacityOutput matchesOut;  // [M,2] int64
  FixedCapacityOutput scoresOut;   // [M]   float

  // Host staging (normalised kpts; descriptors go straight from the caller).
  std::vector<float> h_kpts[2];
  std::vector<std::int64_t> h_matches;
  std::vector<float> h_scores;

  ~Impl() {
    for (int s = 0; s < 2; ++s) {
      cudaFree(d_kpts[s]);
      cudaFree(d_desc[s]);
    }
    if (stream) cudaStreamDestroy(stream);
    delete context;  // TRT 10: objects are deleted, not destroy()'d.
    delete engine;
    delete runtime;
  }

  bool load() {
    cudaSetDevice(cfg.cuda_device);
    std::ifstream f(cfg.engine_path, std::ios::binary);
    if (!f) {
      std::cerr << "[trt-lg] cannot open engine: " << cfg.engine_path << "\n";
      return false;
    }
    std::vector<char> blob((std::istreambuf_iterator<char>(f)), {});

    runtime = nvinfer1::createInferRuntime(logger);
    if (!runtime) return false;
    engine = runtime->deserializeCudaEngine(blob.data(), blob.size());
    if (!engine) return false;
    context = engine->createExecutionContext();
    if (!context) return false;

    // Capacity from the static input shape [1,K,2].
    const nvinfer1::Dims kd = engine->getTensorShape("kpts0");
    if (kd.nbDims != 3) {
      std::cerr << "[trt-lg] unexpected kpts0 shape\n";
      return false;
    }
    capacity = std::uint32_t(kd.d[1]);

    if (cudaStreamCreate(&stream) != cudaSuccess) return false;
    const std::size_t kptsBytes = std::size_t(capacity) * 2 * sizeof(float);
    const std::size_t descBytes =
        std::size_t(capacity) * kDescDim * sizeof(float);
    for (int s = 0; s < 2; ++s) {
      if (cudaMalloc(&d_kpts[s], kptsBytes) != cudaSuccess) return false;
      if (cudaMalloc(&d_desc[s], descBytes) != cudaSuccess) return false;
      h_kpts[s].resize(std::size_t(capacity) * 2);
    }
    // Outputs at worst case M == K.
    if (!matchesOut.init(std::size_t(capacity) * 2 * sizeof(std::int64_t))) {
      return false;
    }
    if (!scoresOut.init(std::size_t(capacity) * sizeof(float))) return false;
    h_matches.resize(std::size_t(capacity) * 2);
    h_scores.resize(capacity);

    if (!context->setOutputAllocator("matches", &matchesOut)) return false;
    if (!context->setOutputAllocator("scores", &scoresOut)) return false;
    return true;
  }

  // Normalise pixel coords per the LightGlue convention and stage one side.
  // Returns the number of slots actually filled (n truncated to capacity).
  std::size_t stage(int side, const float* kpts_px, const float* desc,
                    std::size_t n, std::uint32_t width, std::uint32_t height) {
    const std::size_t used = std::min<std::size_t>(n, capacity);
    const float shiftX = 0.5f * float(width);
    const float shiftY = 0.5f * float(height);
    const float invScale = 2.0f / float(std::max(width, height));
    std::vector<float>& staged = h_kpts[side];
    std::fill(staged.begin(), staged.end(), 0.f);  // zero-pad unused slots
    for (std::size_t i = 0; i < used; ++i) {
      staged[i * 2] = (kpts_px[i * 2] - shiftX) * invScale;
      staged[i * 2 + 1] = (kpts_px[i * 2 + 1] - shiftY) * invScale;
    }
    cudaMemcpyAsync(d_kpts[side], staged.data(),
                    staged.size() * sizeof(float), cudaMemcpyHostToDevice,
                    stream);
    // Descriptors: copy the used rows, zero the padded tail.
    cudaMemcpyAsync(d_desc[side], desc,
                    used * kDescDim * sizeof(float), cudaMemcpyHostToDevice,
                    stream);
    const std::size_t padFloats = std::size_t(capacity - used) * kDescDim;
    if (padFloats) {
      cudaMemsetAsync(d_desc[side] + used * kDescDim, 0,
                      padFloats * sizeof(float), stream);
    }
    return used;
  }
};

LighterGlueMatcher::LighterGlueMatcher(const LighterGlueConfig& cfg)
    : impl_(std::make_unique<Impl>()) {
  impl_->cfg = cfg;
  if (cfg.engine_path.empty()) return;
  if (!impl_->load()) {
    std::cerr << "[trt-lg] LighterGlue engine failed to load: "
              << cfg.engine_path << "\n";
    // Reset to a clean stub state.
    impl_ = std::make_unique<Impl>();
    impl_->cfg = cfg;
    impl_->cfg.engine_path.clear();
  }
}

LighterGlueMatcher::~LighterGlueMatcher() = default;
LighterGlueMatcher::LighterGlueMatcher(LighterGlueMatcher&&) noexcept = default;
LighterGlueMatcher& LighterGlueMatcher::operator=(LighterGlueMatcher&&) noexcept =
    default;

bool LighterGlueMatcher::loaded() const noexcept {
  return impl_->engine != nullptr;
}

std::uint32_t LighterGlueMatcher::capacity() const noexcept {
  return impl_->capacity;
}

PairMatches LighterGlueMatcher::match(
    const float* kptsA, const float* descA, std::size_t nA,
    std::uint32_t widthA, std::uint32_t heightA, const float* kptsB,
    const float* descB, std::size_t nB, std::uint32_t widthB,
    std::uint32_t heightB) {
  PairMatches result;
  Impl& im = *impl_;
  if (!im.context || nA == 0 || nB == 0) return result;

  const std::size_t usedA = im.stage(0, kptsA, descA, nA, widthA, heightA);
  const std::size_t usedB = im.stage(1, kptsB, descB, nB, widthB, heightB);

  im.context->setTensorAddress("kpts0", im.d_kpts[0]);
  im.context->setTensorAddress("kpts1", im.d_kpts[1]);
  im.context->setTensorAddress("desc0", im.d_desc[0]);
  im.context->setTensorAddress("desc1", im.d_desc[1]);
  if (!im.context->enqueueV3(im.stream)) {
    std::cerr << "[trt-lg] enqueue failed\n";
    return result;
  }
  cudaStreamSynchronize(im.stream);

  // M from the allocator's notifyShape ([M,2]).
  const nvinfer1::Dims& md = im.matchesOut.shape();
  const std::size_t numMatches =
      md.nbDims >= 1 ? std::size_t(std::max<std::int64_t>(md.d[0], 0)) : 0;
  if (numMatches == 0 || numMatches > im.capacity) return result;

  cudaMemcpyAsync(im.h_matches.data(), im.matchesOut.data(),
                  numMatches * 2 * sizeof(std::int64_t),
                  cudaMemcpyDeviceToHost, im.stream);
  cudaMemcpyAsync(im.h_scores.data(), im.scoresOut.data(),
                  numMatches * sizeof(float), cudaMemcpyDeviceToHost,
                  im.stream);
  cudaStreamSynchronize(im.stream);

  result.indices.reserve(numMatches);
  result.scores.reserve(numMatches);
  for (std::size_t m = 0; m < numMatches; ++m) {
    const std::int64_t a = im.h_matches[m * 2];
    const std::int64_t b = im.h_matches[m * 2 + 1];
    const float score = im.h_scores[m];
    // Drop matches into zero-padded slots and below the confidence floor.
    if (a < 0 || b < 0 || std::size_t(a) >= usedA || std::size_t(b) >= usedB ||
        score < im.cfg.min_score) {
      continue;
    }
    result.indices.emplace_back(std::uint32_t(a), std::uint32_t(b));
    result.scores.push_back(score);
  }
  return result;
}

}  // namespace xfeat
}  // namespace okvis

#else  // ---------- stub build (no TensorRT) ----------

namespace okvis {
namespace xfeat {

struct LighterGlueMatcher::Impl {};

LighterGlueMatcher::LighterGlueMatcher(const LighterGlueConfig&)
    : impl_(nullptr) {
  std::cerr << "[trt-lg] built without OKVIS_XFEAT_USE_TENSORRT — stub matcher\n";
}
LighterGlueMatcher::~LighterGlueMatcher() = default;
LighterGlueMatcher::LighterGlueMatcher(LighterGlueMatcher&&) noexcept = default;
LighterGlueMatcher& LighterGlueMatcher::operator=(LighterGlueMatcher&&) noexcept =
    default;
bool LighterGlueMatcher::loaded() const noexcept { return false; }
std::uint32_t LighterGlueMatcher::capacity() const noexcept { return 0; }
PairMatches LighterGlueMatcher::match(const float*, const float*, std::size_t,
                                      std::uint32_t, std::uint32_t,
                                      const float*, const float*, std::size_t,
                                      std::uint32_t, std::uint32_t) {
  return PairMatches();
}

}  // namespace xfeat
}  // namespace okvis

#endif  // OKVIS_XFEAT_USE_TENSORRT
