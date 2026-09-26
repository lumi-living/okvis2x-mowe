/**
 * @file preprocess.h
 * @brief CUDA preprocessing kernel for the XFeat frontend (declaration).
 *
 * Implemented in preprocess.cu, compiled only under OKVIS_XFEAT_USE_TENSORRT.
 * The XFeat contract (see export.py): float pixels in the RAW 0..255 range —
 * NO /255, no mean/std (InstanceNorm inside the net handles scale) — laid out
 * NCHW, single channel, at exactly the engine's (H, W). The kernel also does
 * the full-res → engine-res downscale (bilinear, pixel-centre aligned like
 * OpenCV INTER_LINEAR) so the camera's 1280x800 mono8 goes straight into the
 * 640x384 binding. // T-0111, mowe-nav-kb 03 §resolution
 */
#ifndef OKVIS_XFEAT_PREPROCESS_H_
#define OKVIS_XFEAT_PREPROCESS_H_

extern "C" {

/// uint8 mono (row-pitched, `src_w` x `src_h`, `src_pitch` bytes/row) →
/// float32 [1,dst_h,dst_w] raw 0..255 at `dst`, bilinearly resized. Equal dims
/// degrade to an exact copy. Async on `stream` (cudaStream_t as void*). All
/// pointers are device pointers.
void xfeat_preprocess_resize_u8(const void* src, int src_pitch, int src_w,
                                int src_h, float* dst, int dst_w, int dst_h,
                                void* stream);

}  // extern "C"

#endif  // OKVIS_XFEAT_PREPROCESS_H_
