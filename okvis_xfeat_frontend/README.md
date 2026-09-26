# okvis_xfeat_frontend — XFeat-on-TensorRT feature frontend (skeleton)

Bridges the Mow-e camera stack to OKVIS's feature frontend, replacing BRISK
detect+describe with **XFeat** running on **TensorRT**:

```
mowe::camera::FrameBundle            (mowe_camera_core, no ROS)
   │  per synchronized plane (left/right)
   ▼
map DMABUF / NvBufSurface → CUDA     (mowe_camera/gpu_map.hpp — zero-copy)
   ▼
CUDA preprocess (uint8 mono → bilinear resize → raw 0..255 NCHW float, batch=2)
   ▼
TensorRT XFeat engine (.plan) enqueue                   (TensorRTEngine)
   ▼
okvis::xfeat::FrameFeatures          (keypoints in FULL-RES px: u = (x+0.5)·W/640 − 0.5, v = (y+0.5)·H/384 − 0.5)
   ▼
OKVIS MultiFrame adapter (okvis_frontend, T-0112)
```

This is **not** OKVIS's `USE_NN` path. That path is a Torch **semantic-segmentation**
model (Cityscapes sky/person classes — see `okvis_apps/src/nn_test.cpp`) used to
*mask* unreliable regions. XFeat is a separate, new feature frontend.

## Build

Opt-in from the top-level OKVIS build:

```bash
cmake -C okvis2x-mowe/cmake/mowe-defaults.cmake -S okvis2x-mowe -B build -DUSE_MOWE_XFEAT=ON \
      -Dmowe_camera_core_DIR=<prefix>/lib/cmake/mowe_camera_core
# real inference path (on the Jetson):
#   -DUSE_TENSORRT=ON   (in the submodule; needs CUDA 12.5 + TensorRT 10.3)
```

Or standalone:

```bash
cmake -S okvis2x-mowe/okvis_xfeat_frontend -B build \
      -DCMAKE_PREFIX_PATH=<mowe_camera_core install prefix>
```

`mowe_camera_core` must be installed first (it exports a vanilla, ament-free
CMake package, so OKVIS finds it without ROS).

## Status — wired vs. stubbed

| Piece | State |
|---|---|
| `FrameBundle` → frontend plumbing, engine ownership, per-plane loop | ✅ real, compiles + runs |
| Host-image entry point (`extract_image`, the ROS-subscriber path) | ✅ used by `okvis::Frontend` |
| `mowe_camera::core` link + driver self-registration across the `.so` | ✅ verified on dev box |
| Zero-copy `NvBufSurface → EGL → CUDA` mapping (`gpu_map.cpp`) | ⚠️ written, **needs on-device verification** (JetPack 6.2 headers) |
| CUDA preprocess kernel (`preprocess.cu`, pitched mono u8 → bilinear resize 1280x800→640x384 → raw-0..255 NCHW float, one batch slot per eye) | ✅ on-device, T-0111 (`out/agent/T-0111/device/demo.json`) |
| TensorRT 10.3 named I/O (`getTensorShape/DataType`, `setTensorAddress`, `enqueueV3`), batch=2 stereo, contract check `io_names_match` (`TensorRTEngine.cpp`) | ✅ on-device, T-0111 |
| Device→host readback into `FrameFeatures`: −1 padding strip, score threshold, scale-back to full-res px, unit norm (`FrameFeaturesUtil.hpp`, gtest `test/test_frame_features.cpp`) | ✅ on-device + qemu, T-0111 |
| Sub-pixel keypoints: optional 4th engine output `offsets[B,K,2]` (3×3 heatmap centre of mass, `export.py` T-0114) added to the int32 peak before the scale-back; engines without it still load (`has_offsets()`) | ✅ T-0114 — lifts the fixture-pair epipolar inlier ratio (\|Δv\| ≤ 2 px) from 0.83 to ≥ 0.85 (host prototype 0.96) |
| `xfeat_frontend_demo --engine --left --right --iters --json` (PNG pair, pinned host buffers, mean/p99 per pair) | ✅ 6.5 ms mean / 7.0 ms p99 per stereo pair, 640x384 k1024 FP16, Orin Nano MAXN_SUPER (T-0111) |
| `.plan` engines from `trtexec` (`xfeat_*.plan`, `lighterglue_k512_*.plan`) | ✅ on device (`/home/mowe/agent/engines`, T-0110) |
| OKVIS `MultiFrame` hand-off (detect/describe + float matching) | ✅ ADR-0040 stage A (`okvis_frontend`, `USE_MOWE_XFEAT`) |
| **LighterGlue** k512 split matcher (`LighterGlueMatcher`, static `matches0/mscores0`, `scores>0` validity mask, top-K by score, greatest-priority stream; CPU helpers `LighterGlueUtil.hpp`, gtest `test/test_lighterglue.cpp`) | ✅ on-device, T-0114 (`out/agent/T-0114/device/lg.json`): 186 matches, 5.2 ms mean / 11.6 ms p99, mask proven by garbage-padding A/B |
| `lighterglue_demo --engine --xfeat-engine --left --right --iters --json [--dump csv] [--epi-tol px]` (XFeat → LighterGlue vs mutual-NN on the fixture pair, epipolar/disparity inlier ratios, padding-mask A/B) | ✅ T-0114 |
| Place recognition (DBoW replacement) | ❌ ADR-0040 issue #5 (DINOv2/FAISS); loop closures disabled with XFeat |

Without `USE_TENSORRT` the engine runs in **stub mode**: the pipeline executes
end-to-end but emits empty features (useful for wiring/timing the capture path).

## The OKVIS hand-off (ADR-0040 stage A–C, done)

`okvis::Frontend` (built with `USE_MOWE_XFEAT`, enabled via
`frontend_parameters.xfeat` in the OKVIS yaml — see
`config/mowe/okvis2-xfeat.yaml`):

1. **Detect/describe** — `detectAndDescribeXFeat` runs one engine per camera
   (TRT contexts are not thread-safe) and stores keypoints + CV_32F 64-D
   descriptors via `resetKeypoints`/`resetDescriptors`.
2. **Metric** — `descriptorDist()` dispatches BRISK Hamming ↔ cosine distance
   (1 − dot); `matching_threshold` is a cosine distance with XFeat.
3. **Pair matching** — `matchStereo` (L↔R) runs brute-force mutual-NN cosine
   first and calls `LighterGlueMatcher` only when that found fewer than
   `stereo_min_nn_matches` (default 40) pairs, for the still-unmatched
   keypoints (KB 02 decision table, T-0114). `matchMotionStereo` uses it for
   the top-`motion_stereo_top_n` overlap frames. OKVIS's triangulation
   validation + landmark bookkeeping run unchanged on the proposals. Empty
   `lighterglue_engine` = cosine NN only. `frontend_stats.json` counts
   `stereo_lighterglue_fallbacks`.
4. **Place recognition** — still open (issue #5): the DBoW2 vocabulary is
   BRISK-trained, so multi-session + loop-closure paths are disabled under
   XFeat until DINOv2/FAISS lands.
