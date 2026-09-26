/**
 * @file LighterGlueMatcher.cpp
 * @brief LighterGlue TensorRT pair matcher. Real impl under
 *        OKVIS_XFEAT_USE_TENSORRT (TensorRT 10.3), stub otherwise.
 *
 * All engine I/O is static-shape (export.py, T-0109): mutual-NN + threshold
 * run in-graph and the result is matches0[1,K] / mscores0[1,K], so no output
 * allocator is needed. Padding slots carry score -1 and are masked inside the
 * engine (KB 03 §padding); LighterGlueUtil.hpp does the CPU side and is
 * unit-tested without CUDA. // T-0114
 */
#include "okvis/xfeat/LighterGlueMatcher.hpp"

#include <iostream>

#include "okvis/xfeat/LighterGlueUtil.hpp"

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

}  // namespace

struct LighterGlueMatcher::Impl {
  LighterGlueConfig cfg;
  Logger logger;
  nvinfer1::IRuntime* runtime = nullptr;
  nvinfer1::ICudaEngine* engine = nullptr;
  nvinfer1::IExecutionContext* context = nullptr;
  cudaStream_t stream = nullptr;
  std::uint32_t capacity = 0;  // K from the engine bindings.
  bool ioMatch = false;

  // Device slates, per side: [1,K,2] / [1,K,64] / [1,K]; outputs [1,K].
  float* d_kpts[2] = {nullptr, nullptr};
  float* d_desc[2] = {nullptr, nullptr};
  float* d_scores[2] = {nullptr, nullptr};
  std::int32_t* d_matches = nullptr;
  float* d_mscores = nullptr;

  // Pinned host staging (page-locked → async DMA on the stream).
  StagedSet staged[2];
  float* h_desc[2] = {nullptr, nullptr};
  std::int32_t* h_matches = nullptr;
  float* h_mscores = nullptr;

  ~Impl() {
    for (int s = 0; s < 2; ++s) {
      cudaFree(d_kpts[s]);
      cudaFree(d_desc[s]);
      cudaFree(d_scores[s]);
      cudaFreeHost(h_desc[s]);
    }
    cudaFree(d_matches);
    cudaFree(d_mscores);
    cudaFreeHost(h_matches);
    cudaFreeHost(h_mscores);
    if (stream) cudaStreamDestroy(stream);
    delete context;  // TRT 10: objects are deleted, not destroy()'d.
    delete engine;
    delete runtime;
  }

  bool checkIo() const {
    struct Want { const char* name; nvinfer1::DataType dt; int rank; };
    const Want want[] = {
        {"kpts0", nvinfer1::DataType::kFLOAT, 3}, {"kpts1", nvinfer1::DataType::kFLOAT, 3},
        {"desc0", nvinfer1::DataType::kFLOAT, 3}, {"desc1", nvinfer1::DataType::kFLOAT, 3},
        {"scores0", nvinfer1::DataType::kFLOAT, 2}, {"scores1", nvinfer1::DataType::kFLOAT, 2},
        {"matches0", nvinfer1::DataType::kINT32, 2}, {"mscores0", nvinfer1::DataType::kFLOAT, 2}};
    if (engine->getNbIOTensors() != int(sizeof(want) / sizeof(want[0]))) return false;
    for (const Want& w : want) {
      const nvinfer1::Dims d = engine->getTensorShape(w.name);
      if (d.nbDims != w.rank || engine->getTensorDataType(w.name) != w.dt) return false;
      if (d.d[1] != std::int64_t(capacity)) return false;
    }
    return true;
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

    const nvinfer1::Dims kd = engine->getTensorShape("kpts0");
    if (kd.nbDims != 3) {
      std::cerr << "[trt-lg] unexpected kpts0 shape\n";
      return false;
    }
    capacity = std::uint32_t(kd.d[1]);
    ioMatch = checkIo();
    if (!ioMatch) {
      std::cerr << "[trt-lg] engine I/O does not match the export.py contract "
                   "(kpts/desc/scores in, matches0/mscores0 out)\n";
      return false;
    }

    // Same priority as the extractor: VIO must not be starved by perception
    // engines (KB 03 §streams). greatest = numerically lowest.
    int lo = 0, hi = 0;
    cudaDeviceGetStreamPriorityRange(&lo, &hi);
    if (cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, hi) != cudaSuccess) return false;

    const std::size_t K = capacity;
    const std::size_t descBytes = K * kDescDim * sizeof(float);
    for (int s = 0; s < 2; ++s) {
      if (cudaMalloc(&d_kpts[s], K * 2 * sizeof(float)) != cudaSuccess) return false;
      if (cudaMalloc(&d_desc[s], descBytes) != cudaSuccess) return false;
      if (cudaMalloc(&d_scores[s], K * sizeof(float)) != cudaSuccess) return false;
      if (cudaMallocHost(reinterpret_cast<void**>(&h_desc[s]), descBytes) != cudaSuccess) return false;
    }
    if (cudaMalloc(&d_matches, K * sizeof(std::int32_t)) != cudaSuccess) return false;
    if (cudaMalloc(&d_mscores, K * sizeof(float)) != cudaSuccess) return false;
    if (cudaMallocHost(reinterpret_cast<void**>(&h_matches), K * sizeof(std::int32_t)) != cudaSuccess) return false;
    if (cudaMallocHost(reinterpret_cast<void**>(&h_mscores), K * sizeof(float)) != cudaSuccess) return false;

    context->setTensorAddress("kpts0", d_kpts[0]);
    context->setTensorAddress("kpts1", d_kpts[1]);
    context->setTensorAddress("desc0", d_desc[0]);
    context->setTensorAddress("desc1", d_desc[1]);
    context->setTensorAddress("scores0", d_scores[0]);
    context->setTensorAddress("scores1", d_scores[1]);
    context->setTensorAddress("matches0", d_matches);
    context->setTensorAddress("mscores0", d_mscores);
    return true;
  }

  // Stage one side: top-K by score, normalised coords, validity via scores
  // (-1 in unused slots), descriptors gathered into pinned memory, then
  // uploaded on the stream. Unused descriptor rows are zeroed once per call
  // (cheap, K*64 floats) so a stale row can never be read even if masking
  // were ever wrong.
  void stage(int side, const float* kpts_px, const float* desc, const float* scores,
             std::size_t n, std::uint32_t width, std::uint32_t height) {
    StagedSet& st = staged[side];
    stage_set(kpts_px, scores, n, width, height, capacity, st);
    std::memset(h_desc[side], 0, std::size_t(capacity) * kDescDim * sizeof(float));
    for (std::size_t i = 0; i < st.used; ++i) {
      std::memcpy(h_desc[side] + i * kDescDim, desc + std::size_t(st.src_index[i]) * kDescDim,
                  kDescDim * sizeof(float));
    }
    cudaMemcpyAsync(d_kpts[side], st.kpts_norm.data(), st.kpts_norm.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_scores[side], st.scores.data(), st.scores.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_desc[side], h_desc[side], std::size_t(capacity) * kDescDim * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
  }
};

LighterGlueMatcher::LighterGlueMatcher(const LighterGlueConfig& cfg)
    : impl_(std::make_unique<Impl>()) {
  impl_->cfg = cfg;
  if (cfg.engine_path.empty()) return;
  if (!impl_->load()) {
    std::cerr << "[trt-lg] LighterGlue engine failed to load: " << cfg.engine_path << "\n";
    impl_ = std::make_unique<Impl>();  // clean stub state
    impl_->cfg = cfg;
    impl_->cfg.engine_path.clear();
  }
}

LighterGlueMatcher::~LighterGlueMatcher() = default;
LighterGlueMatcher::LighterGlueMatcher(LighterGlueMatcher&&) noexcept = default;
LighterGlueMatcher& LighterGlueMatcher::operator=(LighterGlueMatcher&&) noexcept = default;

bool LighterGlueMatcher::loaded() const noexcept { return impl_->engine != nullptr; }
std::uint32_t LighterGlueMatcher::capacity() const noexcept { return impl_->capacity; }
bool LighterGlueMatcher::io_names_match() const noexcept { return impl_->ioMatch; }

PairMatches LighterGlueMatcher::match(
    const float* kptsA, const float* descA, const float* scoresA, std::size_t nA,
    std::uint32_t widthA, std::uint32_t heightA,
    const float* kptsB, const float* descB, const float* scoresB, std::size_t nB,
    std::uint32_t widthB, std::uint32_t heightB) {
  PairMatches result;
  Impl& im = *impl_;
  if (!im.context || nA == 0 || nB == 0) return result;

  im.stage(0, kptsA, descA, scoresA, nA, widthA, heightA);
  im.stage(1, kptsB, descB, scoresB, nB, widthB, heightB);
  if (!im.context->enqueueV3(im.stream)) {
    std::cerr << "[trt-lg] enqueue failed\n";
    return result;
  }
  cudaMemcpyAsync(im.h_matches, im.d_matches, im.capacity * sizeof(std::int32_t),
                  cudaMemcpyDeviceToHost, im.stream);
  cudaMemcpyAsync(im.h_mscores, im.d_mscores, im.capacity * sizeof(float),
                  cudaMemcpyDeviceToHost, im.stream);
  if (cudaStreamSynchronize(im.stream) != cudaSuccess) {
    std::cerr << "[trt-lg] stream sync failed\n";
    return result;
  }

  DecodedMatches d = decode_matches(im.h_matches, im.h_mscores, im.staged[0], im.staged[1],
                                    im.cfg.min_score);
  result.indices = std::move(d.indices);
  result.scores = std::move(d.scores);
  result.staged_a = std::uint32_t(im.staged[0].used);
  result.staged_b = std::uint32_t(im.staged[1].used);
  result.masked_a = std::uint32_t(im.staged[0].masked_slots());
  result.masked_b = std::uint32_t(im.staged[1].masked_slots());
  return result;
}

}  // namespace xfeat
}  // namespace okvis

#else  // ---------- stub build (no TensorRT) ----------

namespace okvis {
namespace xfeat {

struct LighterGlueMatcher::Impl {};

LighterGlueMatcher::LighterGlueMatcher(const LighterGlueConfig&) : impl_(nullptr) {
  std::cerr << "[trt-lg] built without OKVIS_XFEAT_USE_TENSORRT — stub matcher\n";
}
LighterGlueMatcher::~LighterGlueMatcher() = default;
LighterGlueMatcher::LighterGlueMatcher(LighterGlueMatcher&&) noexcept = default;
LighterGlueMatcher& LighterGlueMatcher::operator=(LighterGlueMatcher&&) noexcept = default;
bool LighterGlueMatcher::loaded() const noexcept { return false; }
std::uint32_t LighterGlueMatcher::capacity() const noexcept { return 0; }
bool LighterGlueMatcher::io_names_match() const noexcept { return false; }
PairMatches LighterGlueMatcher::match(const float*, const float*, const float*, std::size_t,
                                      std::uint32_t, std::uint32_t, const float*, const float*,
                                      const float*, std::size_t, std::uint32_t, std::uint32_t) {
  return PairMatches();
}

}  // namespace xfeat
}  // namespace okvis

#endif  // OKVIS_XFEAT_USE_TENSORRT
