# okvis_xfeat_frontend — XFeat-on-TensorRT feature frontend (skeleton)

Bridges the Mow-e camera stack to OKVIS's feature frontend, replacing BRISK
detect+describe with **XFeat** running on **TensorRT**:

```
mowe::camera::FrameBundle            (mowe_camera_core, no ROS)
   │  per synchronized plane (left/right)
   ▼
map DMABUF / NvBufSurface → CUDA     (mowe_camera/gpu_map.hpp — zero-copy)
   ▼
CUDA preprocess (uint8 mono → padded NCHW float)        [TODO kernel]
   ▼
TensorRT XFeat engine (.plan) enqueue                   (TensorRTEngine)
   ▼
okvis::xfeat::FrameFeatures          → OKVIS MultiFrame adapter [next workstream]
```

This is **not** OKVIS's `USE_NN` path. That path is a Torch **semantic-segmentation**
model (Cityscapes sky/person classes — see `okvis_apps/src/nn_test.cpp`) used to
*mask* unreliable regions. XFeat is a separate, new feature frontend.

## Build

Opt-in from the top-level OKVIS build:

```bash
cmake -S okvis2-mowe -B build -DUSE_MOWE_XFEAT=ON \
      -Dmowe_camera_core_DIR=<prefix>/lib/cmake/mowe_camera_core
# real inference path (on the Jetson):
#   -DUSE_TENSORRT=ON   (in the submodule; needs CUDA 12.5 + TensorRT 10.3)
```

Or standalone:

```bash
cmake -S okvis2-mowe/okvis_xfeat_frontend -B build \
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
| CUDA preprocess kernel (`preprocess.cu`, mono u8 → raw-0..255 NCHW float) | ✅ on-device (demo) |
| TensorRT 10 binding introspection + `enqueueV3` (`TensorRTEngine.cpp`) | ✅ on-device (demo, TRT 10.3) |
| Device→host readback into `FrameFeatures` (score threshold filter) | ✅ |
| `.plan` engines from `trtexec` (`xfeat.plan`, `lighterglue.plan`) | ✅ on device (`/opt/mowe/onnx`) |
| OKVIS `MultiFrame` hand-off (detect/describe + float matching) | ✅ ADR-0040 stage A (`okvis_frontend`, `USE_MOWE_XFEAT`) |
| **LighterGlue** pair matcher (`LighterGlueMatcher`, DDS outputs) | ✅ stage B/C — stereo + top-overlap motion stereo |
| Place recognition (DBoW replacement) | ❌ ADR-0040 issue #5 (DINOv2/FAISS); loop closures disabled with XFeat |

Without `USE_TENSORRT` the engine runs in **stub mode**: the pipeline executes
end-to-end but emits empty features (useful for wiring/timing the capture path).

## The OKVIS hand-off (ADR-0040 stage A–C, done)

`okvis::Frontend` (built with `USE_MOWE_XFEAT`, enabled via
`frontend_parameters.xfeat` in the OKVIS yaml — see
`config/ov9281_sch16t_xfeat.yaml`):

1. **Detect/describe** — `detectAndDescribeXFeat` runs one engine per camera
   (TRT contexts are not thread-safe) and stores keypoints + CV_32F 64-D
   descriptors via `resetKeypoints`/`resetDescriptors`.
2. **Metric** — `descriptorDist()` dispatches BRISK Hamming ↔ cosine distance
   (1 − dot); `matching_threshold` is a cosine distance with XFeat.
3. **Pair matching** — `LighterGlueMatcher` proposes mutual-NN pairs for
   `matchStereo` (L↔R) and the top-`motion_stereo_top_n` overlap frames in
   `matchMotionStereo`; OKVIS's triangulation validation + landmark
   bookkeeping run unchanged on the proposals. Empty `lighterglue_engine`
   falls back to brute-force cosine NN.
4. **Place recognition** — still open (issue #5): the DBoW2 vocabulary is
   BRISK-trained, so multi-session + loop-closure paths are disabled under
   XFeat until DINOv2/FAISS lands.
