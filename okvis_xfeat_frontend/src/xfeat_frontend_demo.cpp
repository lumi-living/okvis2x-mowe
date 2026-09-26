/**
 * @file xfeat_frontend_demo.cpp
 * @brief End-to-end demo of the XFeat frontend without ROS, two modes:
 *
 *   File mode (T-0111 acceptance — the first real GPU run of mow-e code):
 *     xfeat_frontend_demo --engine x.plan --left L.png --right R.png
 *                         [--iters 200] [--warmup 10] [--json out.json]
 *     Loads the PNG pair (mono8, any size; resized on the GPU to the engine's
 *     input), page-locks the host buffers, runs N batch-2 iterations, and
 *     reports counts, unit-norm check, padding strip, engine I/O contract and
 *     mean/p99 wall time per stereo pair excluding the warm-up iterations.
 *
 *   Camera mode (bring-up):
 *     xfeat_frontend_demo <camera_type> <device> [engine.plan] [viz_out_dir]
 *     e.g. xfeat_frontend_demo arducam_ov9281 /dev/video0 xfeat.plan
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "okvis/xfeat/FrameFeaturesUtil.hpp"

#include "mowe_camera/camera.hpp"
#include "mowe_camera/factory.hpp"
#include "mowe_camera/frame.hpp"

#include "okvis/xfeat/XFeatFrontend.hpp"

#ifdef MOWE_XFEAT_DEMO_OPENCV
#include <opencv2/opencv.hpp>

// Draw the per-stream keypoints on their frames, concatenate L|R, and either
// save a PNG (headless) or imshow (when $DISPLAY is set). Keypoints are in the
// per-eye engine resolution, which equals the plane size, so they map 1:1.
// Returns false when the user asked to quit (q/ESC in the live window).
static bool visualize(const mowe::camera::FrameBundle& bundle,
                      const okvis::xfeat::FrameFeatures& feats,
                      const std::string& out_dir) {
  std::vector<cv::Mat> tiles;
  const auto planes = bundle.planes();
  for (std::size_t i = 0; i < planes.size(); ++i) {
    const auto& p = planes[i];
    if (!p.data) continue;
    // Wrap the GREY plane respecting its row stride (stereo split shares one buf).
    cv::Mat gray(p.height, p.width, CV_8UC1,
                 const_cast<std::uint8_t*>(p.data), p.stride_bytes);
    cv::Mat bgr;
    cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
    if (i < feats.streams.size()) {
      const auto& s = feats.streams[i];
      for (const auto& kp : s.keypoints_px) {
        const cv::Point c(int(kp.u), int(kp.v));
        cv::circle(bgr, c, 3, cv::Scalar(0, 0, 0), 2);      // dark halo
        cv::circle(bgr, c, 2, cv::Scalar(0, 255, 0), cv::FILLED);
      }
      cv::putText(bgr, p.stream_id + ": " + std::to_string(s.size()) + " kpts",
                  {10, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.8,
                  {0, 255, 255}, 2);
    }
    tiles.push_back(bgr);
  }
  if (tiles.empty()) return true;
  cv::Mat canvas;
  cv::hconcat(tiles, canvas);
  if (!out_dir.empty()) {
    char name[64];
    std::snprintf(name, sizeof(name), "/xfeat_%06ld.png",
                  static_cast<long>(bundle.sequence()));
    cv::imwrite(out_dir + name, canvas);
    return true;
  }
  cv::imshow("xfeat (L|R)", canvas);
  const int key = cv::waitKey(1);     // pumps the GUI event loop; required
  return !(key == 27 || key == 'q');  // ESC / q quits
}
#endif  // MOWE_XFEAT_DEMO_OPENCV

#ifdef MOWE_XFEAT_DEMO_OPENCV
// Page-aligned copy of a mono8 image so cudaHostRegister pins whole pages.
struct PinnedImage {
  std::uint8_t* data = nullptr;
  std::uint32_t w = 0, h = 0;
  std::size_t bytes = 0;
  explicit PinnedImage(const cv::Mat& m) : w(m.cols), h(m.rows) {
    bytes = ((std::size_t(w) * h + 4095) / 4096) * 4096;
    data = static_cast<std::uint8_t*>(std::aligned_alloc(4096, bytes));
    for (int r = 0; r < m.rows; ++r) std::memcpy(data + std::size_t(r) * w, m.ptr(r), w);
  }
  ~PinnedImage() { std::free(data); }
};

static int run_file_mode(const std::map<std::string, std::string>& a) {
  auto get = [&](const char* k, const char* def = "") {
    auto it = a.find(k);
    return it == a.end() ? std::string(def) : it->second;
  };
  const std::string engine = get("--engine"), left = get("--left"), right = get("--right");
  const int iters = std::atoi(get("--iters", "200").c_str());
  const int warmup = std::atoi(get("--warmup", "10").c_str());
  const std::string json = get("--json");
  if (engine.empty() || left.empty() || right.empty() || iters <= warmup) {
    std::cerr << "file mode needs --engine --left --right (and --iters > --warmup)\n";
    return 1;
  }
  cv::Mat L = cv::imread(left, cv::IMREAD_GRAYSCALE);
  cv::Mat R = cv::imread(right, cv::IMREAD_GRAYSCALE);
  if (L.empty() || R.empty() || L.size() != R.size()) {
    std::cerr << "cannot read " << left << " / " << right << " as an equal-size mono pair\n";
    return 2;
  }
  PinnedImage pl(L), pr(R);

  okvis::xfeat::XFeatConfig cfg;
  cfg.engine_path = engine;
  okvis::xfeat::XFeatFrontend fe(cfg);
  if (!fe.engine_loaded()) {
    std::cerr << "engine failed to load: " << engine << "\n";
    return 3;
  }
  const bool pinned = fe.register_host_buffer(pl.data, pl.bytes) &&
                      fe.register_host_buffer(pr.data, pr.bytes);
  std::uint32_t ew = 0, eh = 0;
  fe.input_dims(ew, eh);
  const okvis::xfeat::ResizeScale sc = okvis::xfeat::resize_scale(pl.w, pl.h, ew, eh);
  std::cout << "engine " << ew << "x" << eh << " batch " << fe.batch() << " k " << fe.max_keypoints()
            << " io_names_match " << fe.io_names_match() << " pinned " << pinned
            << " scale (" << sc.sx << ", " << sc.sy << ")\n";
  for (const auto& t : fe.tensor_summary()) std::cout << "  " << t << "\n";

  std::vector<double> ms;
  okvis::xfeat::FrameFeatures ff;
  for (int i = 0; i < iters; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    ff = fe.extract_pair(pl.data, pl.w, pr.data, pr.w, pl.w, pl.h);
    const auto t1 = std::chrono::steady_clock::now();
    if (i >= warmup) ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  fe.unregister_host_buffer(pl.data);
  fe.unregister_host_buffer(pr.data);

  // Padding path on the real engine: a textured fixture can fill all K slots
  // with local maxima (no -1 rows), so also run a flat mono8 pair — almost no
  // detections → nearly every slot is a -1 padding row that must be stripped.
  const std::uint32_t K = fe.max_keypoints();
  std::vector<std::uint8_t> flat(std::size_t(pl.w) * pl.h, 128);
  const okvis::xfeat::FrameFeatures blank = fe.extract_pair(flat.data(), pl.w, flat.data(), pl.w, pl.w, pl.h);
  const auto& bl = blank.streams.at(0);
  const bool blank_ok = bl.padding_rows > 0 && bl.padding_rows <= K &&
                        std::none_of(bl.scores.begin(), bl.scores.end(), [](float v) { return v < 0.f; });

  std::vector<double> sorted = ms;
  std::sort(sorted.begin(), sorted.end());
  const double mean = std::accumulate(ms.begin(), ms.end(), 0.0) / double(ms.size());
  const double p99 = sorted[std::min(sorted.size() - 1, std::size_t(std::ceil(0.99 * sorted.size())) - 1)];
  const auto& l = ff.streams.at(0);
  const auto& r = ff.streams.at(1);
  const float dev = std::max(okvis::xfeat::max_unit_norm_deviation(l.descriptors),
                             okvis::xfeat::max_unit_norm_deviation(r.descriptors));
  // Every -1 row was removed and no negative score survived — on both eyes.
  auto stripped_ok = [K](const okvis::xfeat::StreamFeatures& s) {
    return s.size() + s.padding_rows <= K &&
           std::none_of(s.scores.begin(), s.scores.end(), [](float v) { return v < 0.f; });
  };
  auto in_bounds = [&](const okvis::xfeat::StreamFeatures& s) {
    return std::all_of(s.keypoints_px.begin(), s.keypoints_px.end(), [&](const okvis::xfeat::Keypoint& k) {
      return k.u >= -0.5f && k.u <= pl.w - 0.5f && k.v >= -0.5f && k.v <= pl.h - 0.5f;
    });
  };
  const int padding_ok = stripped_ok(l) && stripped_ok(r) && blank_ok;
  const int unit_norm_ok = dev < 0.02f;  // FP16 descriptor head → ~1e-3 typical
  const int coords_ok = in_bounds(l) && in_bounds(r);

  std::string out = "{\n";
  auto kv = [&](const char* k, const std::string& v, bool last = false) {
    out += std::string("  \"") + k + "\": " + v + (last ? "\n" : ",\n");
  };
  kv("engine", "\"" + engine + "\"");
  kv("left", "\"" + left + "\"");
  kv("right", "\"" + right + "\"");
  kv("image_width", std::to_string(pl.w));
  kv("image_height", std::to_string(pl.h));
  kv("engine_width", std::to_string(ew));
  kv("engine_height", std::to_string(eh));
  kv("engine_batch", std::to_string(fe.batch()));
  kv("engine_topk", std::to_string(K));
  kv("scale_u", std::to_string(sc.sx));
  kv("scale_v", std::to_string(sc.sy));
  kv("io_names_match", std::to_string(int(fe.io_names_match())));
  {
    std::string arr = "[";
    bool first = true;
    for (const auto& t : fe.tensor_summary()) { arr += (first ? "\"" : ", \"") + t + "\""; first = false; }
    kv("tensors", arr + "]");
  }
  kv("host_buffers_pinned", std::to_string(int(pinned)));
  kv("left_keypoints", std::to_string(l.size()));
  kv("right_keypoints", std::to_string(r.size()));
  kv("left_padding_rows", std::to_string(l.padding_rows));
  kv("right_padding_rows", std::to_string(r.padding_rows));
  // Rows that were valid local maxima but below cfg.score_threshold (0.05).
  kv("left_below_threshold_rows", std::to_string(K - l.size() - l.padding_rows));
  kv("right_below_threshold_rows", std::to_string(K - r.size() - r.padding_rows));
  kv("blank_image_keypoints", std::to_string(bl.size()));
  kv("blank_image_padding_rows", std::to_string(bl.padding_rows));
  kv("padding_rows_stripped", std::to_string(padding_ok));
  kv("keypoints_in_bounds", std::to_string(coords_ok));
  kv("descriptor_dim", std::to_string(okvis::xfeat::kDescriptorDim));
  kv("max_unit_norm_deviation", std::to_string(dev));
  kv("unit_norm_ok", std::to_string(unit_norm_ok));
  kv("iters", std::to_string(iters));
  kv("warmup", std::to_string(warmup));
  kv("mean_ms_pair", std::to_string(mean));
  kv("p99_ms_pair", std::to_string(p99));
  kv("min_ms_pair", std::to_string(sorted.front()));
  kv("max_ms_pair", std::to_string(sorted.back()), true);
  out += "}\n";
  std::cout << out;
  if (!json.empty()) {
    std::ofstream f(json);
    if (!f) { std::cerr << "cannot write " << json << "\n"; return 4; }
    f << out;
  }
  return 0;
}
#endif  // MOWE_XFEAT_DEMO_OPENCV

int main(int argc, char** argv) {
  if (argc >= 2 && argv[1][0] == '-') {  // file mode: --key value pairs
    std::map<std::string, std::string> a;
    for (int i = 1; i + 1 < argc; i += 2) a[argv[i]] = argv[i + 1];
#ifdef MOWE_XFEAT_DEMO_OPENCV
    return run_file_mode(a);
#else
    std::cerr << "file mode needs OpenCV (imgcodecs) at build time\n";
    return 1;
#endif
  }
  if (argc < 3) {
    std::cerr << "usage: " << argv[0]
              << " <camera_type> <device> [engine.plan] [viz_out_dir]\n"
              << "       " << argv[0]
              << " --engine x.plan --left L.png --right R.png [--iters N] [--warmup N] [--json out.json]\n";
    return 1;
  }

  mowe::camera::CameraConfig cam_cfg;
  cam_cfg.type = argv[1];
  cam_cfg.device = argv[2];
  // OV9281 jetvariety stereo: both eyes arrive as one 2560x800 mono frame; the
  // driver splits to left/right planes (1280x800 each) in place — matches the
  // per-eye XFeat engine. See mowe-camera/config/ov9281_stereo.yaml (ADR-0005).
  if (cam_cfg.type == "arducam_ov9281") {
    cam_cfg.width = 2560;
    cam_cfg.height = 800;
    cam_cfg.pixel_format = "GREY";
    cam_cfg.params["stereo"] = "true";
    cam_cfg.params["buffer_count"] = "4";
    cam_cfg.params["fps"] = "50.0";
  }

  okvis::xfeat::XFeatConfig fe_cfg;  // input dims auto-read from the engine
  if (argc >= 4) fe_cfg.engine_path = argv[3];
  // Tune the detection floor on-device without a rebuild: raise to drop the
  // low-texture noise band, lower for denser keypoints.
  if (const char* th = std::getenv("MOWE_XFEAT_SCORE_THRESH")) {
    fe_cfg.score_threshold = std::strtof(th, nullptr);
    std::cout << "score_threshold = " << fe_cfg.score_threshold << "\n";
  }
  const std::string viz_out = (argc >= 5) ? argv[4] : "";  // PNG dir (optional)

  try {
    auto camera = mowe::camera::CameraFactory::create(cam_cfg);
    okvis::xfeat::XFeatFrontend frontend(fe_cfg);

    if (camera->start() != mowe::camera::CaptureStatus::Ok) {
      std::cerr << "camera start failed\n";
      return 2;
    }

    long max_frames = 100;  // headless / PNG-save: bounded run
#ifdef MOWE_XFEAT_DEMO_OPENCV
    const bool live = viz_out.empty() && std::getenv("DISPLAY") != nullptr;
    if (live) {
      max_frames = -1;  // live window: run until q/ESC
      std::cout << "live view — press q or ESC in the window to quit\n";
    }
#endif
    bool diagged = false;  // print the coord diagnostic on the first good frame
    for (long i = 0; max_frames < 0 || i < max_frames; ++i) {
      mowe::camera::FrameBundle bundle;
      if (camera->capture(std::chrono::milliseconds(200), bundle) !=
          mowe::camera::CaptureStatus::Ok) {
        std::cerr << "capture timeout/error\n";
        continue;
      }
      auto feats = frontend.extract(bundle);
      std::cout << "seq " << feats.sequence << " t " << feats.timestamp_ns
                << "ns:";
      for (const auto& s : feats.streams) {
        std::cout << " [" << s.stream_id << "] " << s.size() << " kpts";
      }
      std::cout << (frontend.engine_loaded() ? "" : "  (stub)") << "\n";

      // One-shot coordinate diagnostic: where do keypoints actually land vs the
      // plane? If the bbox is tiny / off the plane size, it's a coord-space bug,
      // not a rendering one.
      if (!diagged) {
        diagged = true;
        const auto planes = bundle.planes();
        for (std::size_t k = 0; k < feats.streams.size(); ++k) {
          const auto& s = feats.streams[k];
          if (s.keypoints_px.empty()) continue;
          float umin = 1e9f, umax = -1e9f, vmin = 1e9f, vmax = -1e9f;
          long in_bounds = 0;
          const auto& pl = planes[k];
          for (const auto& kp : s.keypoints_px) {
            umin = std::min(umin, kp.u); umax = std::max(umax, kp.u);
            vmin = std::min(vmin, kp.v); vmax = std::max(vmax, kp.v);
            if (kp.u >= 0 && kp.u < pl.width && kp.v >= 0 && kp.v < pl.height)
              ++in_bounds;
          }
          std::cout << "  [diag " << s.stream_id << "] plane "
                    << pl.width << "x" << pl.height << "  u[" << umin << ","
                    << umax << "] v[" << vmin << "," << vmax << "]  in-bounds "
                    << in_bounds << "/" << s.keypoints_px.size() << "\n";
        }
      }
#ifdef MOWE_XFEAT_DEMO_OPENCV
      if (!visualize(bundle, feats, viz_out)) break;
#else
      (void)viz_out;
#endif
    }
    camera->stop();
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 3;
  }
  return 0;
}
