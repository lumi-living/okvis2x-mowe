/**
 * @file preprocess.h
 * @brief CUDA preprocessing kernels for the XFeat frontend (declarations).
 *
 * Implemented in preprocess.cu, compiled only under OKVIS_XFEAT_USE_TENSORRT.
 * The XFeat contract (see export.py): float pixels in the RAW 0..255 range —
 * NO /255, no mean/std (InstanceNorm inside the net handles scale) — laid out
 * NCHW, single channel, at exactly the engine's (H, W).
 */
#ifndef OKVIS_XFEAT_PREPROCESS_H_
#define OKVIS_XFEAT_PREPROCESS_H_

extern "C" {

/// uint8 mono (row-pitched) → float32 NCHW [1,1,H,W], raw 0..255. Async on the
/// given cudaStream_t (void* to keep CUDA out of the C++ headers). `src_pitch`
/// is the source row stride in BYTES (≥ width); device pointers.
void xfeat_preprocess_mono_u8(const void* src, int src_pitch,
                              float* dst, int width, int height, void* stream);

}  // extern "C"

#endif  // OKVIS_XFEAT_PREPROCESS_H_
