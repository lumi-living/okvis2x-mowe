/**
 * @file okvis2x_node_mowe.cpp
 * @brief Mow-e VIO node: owns the OV9281 stereo pair directly via
 *        mowe_camera_core and feeds OKVIS2-X through ThreadedSlam::addImages.
 *
 * Ported from okvis2-mowe `okvis_node_mowe.cpp` (baf246f) onto OKVIS2-X and
 * renamed to follow the OKVIS2-X `okvis2x_node_*` convention.
 *
 * The image hot path has NO ROS in it (ADR-0040 §2): the camera delivers both
 * eyes as ONE hardware-synchronized FrameBundle (single V4L2 frame, single
 * CLOCK_MONOTONIC SOF stamp), so the stereo pair can never split, drop one
 * half, or mis-pair — the failure modes of the two-topic subscriber path. It
 * also sidesteps the mowe_camera lifecycle-activation trap entirely.
 *
 * IMU still arrives over ROS from the sch16t node (small messages; fine over
 * DDS), and pose/visualisation go out via okvis::Publisher. Because the SCH16T
 * node is a LifecycleNode whose launch-side auto configure/activate is
 * unreliable, this node drives the transitions itself at startup (parameter
 * `activate_lifecycle_node`).
 *
 * OKVIS2-X differences vs. the okvis2 version:
 *  - VI(+GNSS) only: no supereight2 submapping, so unlike okvis2x_node_* this
 *    node needs NO `se_config_filename`; it refuses to start when the config
 *    sets `output_parameters.enable_submapping: true`.
 *  - okvis::Publisher takes three ThreadedPublishers (started below).
 *  - `shutdown` service (std_srvs/SetBool) as in okvis2x_node, final BA
 *    (if configured) and the final trajectory CSV in `csv_path`.
 *  - GNSS / wheel odometry are NOT wired yet: see docs/MOWE_TODO.md.
 */
#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <thread>

#include <glog/logging.h>

#include <boost/filesystem.hpp>
#include <opencv2/core.hpp>

#include <okvis/ThreadedPublisher.hpp>
#include <okvis/ThreadedSlam.hpp>
#include <okvis/ViParametersReader.hpp>
#include <okvis/ros2/Publisher.hpp>

#include <lifecycle_msgs/msg/transition.hpp>
#include <lifecycle_msgs/srv/change_state.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_srvs/srv/set_bool.hpp>

#include "mowe_camera/camera.hpp"
#include "mowe_camera/factory.hpp"
#include "mowe_camera/frame.hpp"

static std::atomic_bool shtdown; ///< Shutdown requested?

/// \brief Bench/bring-up counters written to `stats_path` as JSON on shutdown
///        (T-0115; ADR-0042 issue 3 memory budget). Atomics: the capture
///        thread, the IMU callback and the main loop all touch them.
struct RunStats {
  std::atomic<uint64_t> framesCaptured{0};  ///< FrameBundles from the driver
  std::atomic<uint64_t> framesDecimated{0}; ///< skipped by camera_decimation
  std::atomic<uint64_t> framesIngested{0};  ///< addImages() accepted
  std::atomic<uint64_t> framesDropped{0};   ///< addImages() queue-full drop
  std::atomic<uint64_t> framesUnpaired{0};  ///< bundle without both eyes
  std::atomic<uint64_t> imuIngested{0};     ///< addImuMeasurement() accepted
  std::atomic<uint64_t> odomPublished{0};   ///< realtimePredictAndPublish() true
  std::atomic<int> crashes{0};              ///< fatal signals / uncaught exceptions
  std::atomic<int> engineLoaded{0};
  std::atomic<int> cleanExit{0};
  double rssStartMb = 0, rssPeakMb = 0, rssEndMb = 0;
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  std::string path; ///< empty: disabled
};
static RunStats g_stats;

/// \brief Current resident set size from /proc/self/status [MB].
static double rssMb() {
  std::ifstream f("/proc/self/status");
  std::string key;
  long kb = 0;
  while (f >> key) {
    if (key == "VmRSS:") { f >> kb; break; }
    f.ignore(1024, '\n');
  }
  return double(kb) / 1024.0;
}

/// \brief Sample RSS into start/peak/end. Call periodically from the main loop.
static void sampleRss() {
  const double mb = rssMb();
  if (g_stats.rssStartMb == 0) g_stats.rssStartMb = mb;
  g_stats.rssPeakMb = std::max(g_stats.rssPeakMb, mb);
  g_stats.rssEndMb = mb;
}

/// \brief Write the stats JSON (idempotent; also called from the fatal-signal
///        handler, so keep it to plain stream output).
static void writeStats() {
  if (g_stats.path.empty()) return;
  sampleRss();
  const double wallS = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - g_stats.t0).count();
  const uint64_t offered = g_stats.framesCaptured - g_stats.framesDecimated;
  std::ofstream f(g_stats.path);
  f << "{\n"
    << "  \"frames_captured\": " << g_stats.framesCaptured << ",\n"
    << "  \"frames_decimated\": " << g_stats.framesDecimated << ",\n"
    << "  \"frames_offered\": " << offered << ",\n"
    << "  \"frames_ingested\": " << g_stats.framesIngested << ",\n"
    << "  \"frames_dropped\": " << g_stats.framesDropped << ",\n"
    << "  \"frames_unpaired\": " << g_stats.framesUnpaired << ",\n"
    << "  \"frames_ingested_ratio\": "
    << (offered ? double(g_stats.framesIngested) / double(offered) : 0.0) << ",\n"
    << "  \"imu_ingested\": " << g_stats.imuIngested << ",\n"
    << "  \"odom_published\": " << g_stats.odomPublished << ",\n"
    << "  \"engine_loaded\": " << g_stats.engineLoaded << ",\n"
    << "  \"crashes\": " << g_stats.crashes << ",\n"
    << "  \"clean_exit\": " << g_stats.cleanExit << ",\n"
    << "  \"rss_start_mb\": " << g_stats.rssStartMb << ",\n"
    << "  \"rss_peak_mb\": " << g_stats.rssPeakMb << ",\n"
    << "  \"rss_end_mb\": " << g_stats.rssEndMb << ",\n"
    << "  \"rss_growth_mb_per_min\": "
    << (wallS > 1.0 ? (g_stats.rssEndMb - g_stats.rssStartMb) / (wallS / 60.0) : 0.0) << ",\n"
    << "  \"wall_s\": " << wallS << "\n"
    << "}\n";
}

/// \brief Last-ditch: record the crash in the stats file, then die normally.
static void onFatalSignal(int sig) {
  g_stats.crashes = 1;
  writeStats();
  signal(sig, SIG_DFL);
  raise(sig);
}

/// \brief Drive a LifecycleNode through configure -> activate via its
///        change_state service. Best-effort: logs and carries on when a
///        transition is rejected (e.g. the node was activated manually).
static void driveLifecycle(const rclcpp::Node::SharedPtr &node,
                           const std::string &target) {
  using lifecycle_msgs::msg::Transition;
  using lifecycle_msgs::srv::ChangeState;
  auto client = node->create_client<ChangeState>(target + "/change_state");
  if (!client->wait_for_service(std::chrono::seconds(10))) {
    LOG(WARNING) << target << "/change_state not available — is the node up?";
    return;
  }
  for (const std::uint8_t id :
       {Transition::TRANSITION_CONFIGURE, Transition::TRANSITION_ACTIVATE}) {
    auto request = std::make_shared<ChangeState::Request>();
    request->transition.id = id;
    auto future = client->async_send_request(request);
    if (rclcpp::spin_until_future_complete(node, future,
                                           std::chrono::seconds(10)) !=
        rclcpp::FutureReturnCode::SUCCESS) {
      LOG(WARNING) << target << ": lifecycle transition " << int(id)
                   << " timed out";
      return;
    }
    if (!future.get()->success) {
      LOG(WARNING) << target << ": lifecycle transition " << int(id)
                   << " rejected (already active?)";
    }
  }
  LOG(INFO) << target << " driven through configure + activate";
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = 0; // INFO: 0, WARNING: 1, ERROR: 2, FATAL: 3
  FLAGS_colorlogtostderr = 1;

  rclcpp::init(argc, argv);
  std::shared_ptr<rclcpp::Node> node =
      rclcpp::Node::make_shared("okvis2x_node_mowe");

  // Parameters.
  node->declare_parameter("config_filename", "");
  node->declare_parameter("imu_propagated_state_publishing_rate", 0.0);
  node->declare_parameter("camera_type", "arducam_ov9281");
  node->declare_parameter("camera_device", "/dev/video1");
  node->declare_parameter("camera_fps", 50.0);
  node->declare_parameter("camera_decimation", 1);
  node->declare_parameter("activate_lifecycle_node", "/sch16t_imu_node");
  node->declare_parameter("csv_path", "/tmp/");
  // T-0115: bring-up counters + RSS written here as JSON on shutdown ("" = off).
  node->declare_parameter("stats_path", "");

  std::string configFilename;
  node->get_parameter("config_filename", configFilename);
  if (configFilename.empty()) {
    LOG(ERROR) << "ros parameter 'config_filename' not set";
    return EXIT_FAILURE;
  }
  double imu_propagated_state_publishing_rate = 0.0;
  node->get_parameter("imu_propagated_state_publishing_rate",
                      imu_propagated_state_publishing_rate);
  std::string csvPath = "/tmp/";
  node->get_parameter("csv_path", csvPath);
  node->get_parameter("stats_path", g_stats.path);
  if (!g_stats.path.empty()) {
    for (int sig : {SIGSEGV, SIGABRT, SIGBUS, SIGFPE}) signal(sig, onFatalSignal);
  }

  okvis::ViParametersReader viParametersReader(configFilename);
  okvis::ViParameters parameters;
  viParametersReader.getParameters(parameters);
  OKVIS_ASSERT_TRUE(std::runtime_error,
                    parameters.nCameraSystem.numCameras() == 2,
                    "okvis2x_node_mowe drives the OV9281 stereo pair — config "
                    "must declare exactly 2 cameras")
  if (parameters.output.enable_submapping) {
    // This node builds no supereight2 SubmappingInterface (and therefore
    // needs no se2 config). Refuse rather than silently run without maps.
    LOG(ERROR) << "okvis2x_node_mowe is VI(+GNSS) only: set "
                  "output_parameters.enable_submapping: false in "
               << configFilename;
    return EXIT_FAILURE;
  }

  // Publisher (pose, path, match visualisations). OKVIS2-X publishes from
  // three worker threads (wiring as in okvis2x_node.cpp; cross-built and run
  // on the bench in T-0115).
  auto threadedOdometryPublisher =
      std::make_shared<okvis::ThreadedPublisher>(node);
  auto threadedImagePublisher = std::make_shared<okvis::ThreadedPublisher>(node);
  auto threadedPublisher = std::make_shared<okvis::ThreadedPublisher>(node);
  okvis::Publisher publisher(node, threadedOdometryPublisher,
                             threadedImagePublisher, threadedPublisher);

  // DBoW2 vocabulary. BRISK only: with xfeat.use the frontend never loads it
  // (float descriptors, lazy DBoW — T-0112), so its absence is not an error.
  boost::filesystem::path executable(argv[0]);
  const std::string dBowVocDir =
      executable.remove_filename().string() + "/../../share/okvis/resources/";
  std::ifstream infile(dBowVocDir + "/small_voc.yml.gz");
  if (!infile.good()) {
    if (!parameters.frontend.xfeat.use) {
      LOG(ERROR) << "DBoW2 vocabulary " << dBowVocDir
                 << "/small_voc.yml.gz not found.";
      return EXIT_FAILURE;
    }
    LOG(INFO) << "DBoW2 vocabulary not found at " << dBowVocDir
              << " — not needed with xfeat.use (float descriptors)";
  }

  // Default se::SubMapConfig: unused with enable_submapping == false.
  okvis::ThreadedSlam estimator(parameters, dBowVocDir);
  estimator.setBlocking(false);
  // Frontend ctor asserts on a failed engine load, so reaching here with
  // xfeat.use means the TensorRT engine is up.
  g_stats.engineLoaded = estimator.frontend().usingXFeat() ? 1 : 0;
  sampleRss();

  publisher.setBodyTransform(parameters.imu.T_BS);
  publisher.setOdometryPublishingRate(imu_propagated_state_publishing_rate);
  publisher.setupImageTopics(parameters.nCameraSystem);
  estimator.setOptimisedGraphCallback(
      [&publisher](const okvis::State &state,
                   const okvis::TrackingState &trackingState,
                   std::shared_ptr<const okvis::AlignedMap<okvis::StateId, okvis::State>>
                       updatedStates,
                   std::shared_ptr<const okvis::MapPointVector> landmarks) {
        publisher.publishEstimatorUpdate(state, trackingState, updatedStates,
                                         landmarks);
      });

  // Bring the (lifecycle) IMU node up before subscribing — the launch-side
  // auto configure/activate helpers are unreliable.
  std::string lifecycleTarget;
  node->get_parameter("activate_lifecycle_node", lifecycleTarget);
  if (!lifecycleTarget.empty()) {
    driveLifecycle(node, lifecycleTarget);
  }

  // IMU in over ROS: sensor QoS (best-effort) with a deep queue, matching the
  // sch16t publisher; same conversion as okvis::Subscriber::imuCallback.
  auto imuSubscription = node->create_subscription<sensor_msgs::msg::Imu>(
      "imu0", rclcpp::SensorDataQoS().keep_last(1000),
      [&estimator, &publisher](const sensor_msgs::msg::Imu &msg) {
        const okvis::Time timestamp(msg.header.stamp.sec,
                                    msg.header.stamp.nanosec);
        const Eigen::Vector3d acc(msg.linear_acceleration.x,
                                  msg.linear_acceleration.y,
                                  msg.linear_acceleration.z);
        const Eigen::Vector3d gyr(msg.angular_velocity.x,
                                  msg.angular_velocity.y,
                                  msg.angular_velocity.z);
        if (estimator.addImuMeasurement(timestamp, acc, gyr)) ++g_stats.imuIngested;
        if (publisher.realtimePredictAndPublish(timestamp, acc, gyr)) ++g_stats.odomPublished;
      });

  // TODO(mow-e, docs/MOWE_TODO.md): GNSS (NavSatFix -> addGpsMeasurement) and
  // wheel odometry subscriptions go here.

  // Camera in directly via mowe_camera_core (no ROS on the image path).
  mowe::camera::CameraConfig cam_cfg;
  node->get_parameter("camera_type", cam_cfg.type);
  node->get_parameter("camera_device", cam_cfg.device);
  double fps = 50.0;
  node->get_parameter("camera_fps", fps);
  if (cam_cfg.type == "arducam_ov9281") {
    // Both eyes in one 2560x800 GREY frame; the driver splits to left/right
    // planes sharing one buffer + one SOF stamp (ADR-0005/ADR-0012).
    cam_cfg.width = 2560;
    cam_cfg.height = 800;
    cam_cfg.pixel_format = "GREY";
    cam_cfg.params["stereo"] = "true";
    cam_cfg.params["buffer_count"] = "4";
    cam_cfg.params["fps"] = std::to_string(fps);
  }

  try {
    auto camera = mowe::camera::CameraFactory::create(cam_cfg);
    if (camera->start() != mowe::camera::CaptureStatus::Ok) {
      LOG(ERROR) << "camera start failed (" << cam_cfg.type << " on "
                 << cam_cfg.device << ")";
      return EXIT_FAILURE;
    }
    LOG(INFO) << "camera streaming: " << cam_cfg.type << " on "
              << cam_cfg.device << " @ " << fps << " fps (direct ingest)";

    shtdown = false;
    signal(SIGINT, [](int) { shtdown = true; });

    // Shutdown service, as in okvis2x_node: lets a supervisor stop the node
    // cleanly (final BA + CSV) without a signal.
    auto shutdownService = node->create_service<std_srvs::srv::SetBool>(
        "shutdown",
        [](const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
           std::shared_ptr<std_srvs::srv::SetBool::Response> response) {
          (void)request;
          shtdown = true;
          response->success = true;
          response->message = "Requested Shutdown. Starting offline processing.";
        });

    threadedOdometryPublisher->startThread();
    threadedImagePublisher->startThread();
    threadedPublisher->startThread();

    int decimation = 1;
    node->get_parameter("camera_decimation", decimation);
    if (decimation > 1) {
      LOG(INFO) << "feeding every " << decimation << ". frame to the estimator ("
                << fps / decimation << " Hz nominal)";
    }

    // Capture thread: FrameBundle -> addImages. Planes are cloned: the
    // bundle's V4L2 buffer requeues when it goes out of scope, while the
    // estimator queue outlives it. (The GPU zero-copy path — NvBufSurface
    // straight into the XFeat engine — is a follow-up; ADR-0040 §3.)
    std::thread captureThread([&camera, &estimator, decimation]() {
      long timeouts = 0;
      while (!shtdown) {
        mowe::camera::FrameBundle bundle;
        if (camera->capture(std::chrono::milliseconds(200), bundle) !=
            mowe::camera::CaptureStatus::Ok) {
          if (++timeouts % 25 == 0) {
            LOG(WARNING) << "camera capture timeouts: " << timeouts;
          }
          continue;
        }
        ++g_stats.framesCaptured;
        // Sensor-rate decimation: the OV9281 mode runs at 50 fps, more than
        // the Orin Nano VIO pipeline sustains — skip early instead of paying
        // clone+queue for frames ThreadedSlam would drop anyway (sequence-
        // based, so the kept cadence is stable).
        if (decimation > 1 && (bundle.sequence() % decimation) != 0) {
          ++g_stats.framesDecimated;
          continue;
        }
        const auto planes = bundle.planes();
        std::map<size_t, cv::Mat> images;
        for (const auto &plane : planes) {
          if (!plane.data)
            continue;
          // mowe_camera's side-by-side splitter labels the eyes "cam0"/"cam1"
          // (sbs.hpp); "left"/"right" kept for other drivers.
          const size_t camIdx =
              (plane.stream_id == "cam1" || plane.stream_id == "right") ? 1 : 0;
          images[camIdx] = cv::Mat(int(plane.height), int(plane.width), CV_8UC1,
                                   const_cast<std::uint8_t *>(plane.data),
                                   plane.stride_bytes)
                               .clone();
        }
        if (images.size() != 2) {
          if (++g_stats.framesUnpaired % 100 == 1) {
            std::string ids;
            for (const auto &plane : planes) ids += plane.stream_id + " ";
            LOG(WARNING) << "incomplete stereo pair, planes: " << ids
                         << "(" << g_stats.framesUnpaired << " so far)";
          }
          continue; // need the full synchronized pair
        }
        okvis::Time t;
        t.fromNSec(std::uint64_t(bundle.timestamp().count()));
        // false = ThreadedSlam's 2-deep input queue was full (estimator slower
        // than the decimated camera rate) and the oldest frame was dropped.
        if (estimator.addImages(t, images)) ++g_stats.framesIngested;
        else ++g_stats.framesDropped;
      }
      camera->stop();
    });

    // Main loop (same shape as okvis2x_node) + 1 Hz RSS sampling.
    auto nextRss = std::chrono::steady_clock::now();
    while (!shtdown) {
      rclcpp::spin_some(node);
      estimator.processFrame();
      std::map<std::string, cv::Mat> images;
      estimator.display(images);
      publisher.publishImages(images);
      if (std::chrono::steady_clock::now() >= nextRss) {
        sampleRss();
        nextRss += std::chrono::seconds(1);
      }
    }
    captureThread.join();
    imuSubscription.reset();
    // Write once now (clean_exit 0) so the numbers survive a hung final BA,
    // and again after the CSV tail with clean_exit 1.
    writeStats();

    // No state was ever added (e.g. no keypoints — dark frames): final BA has
    // nothing to optimise and writeFinalCsvTrajectory throws map::at. Skip.
    const bool haveStates = estimator.frontend().isInitialized();
    if (parameters.estimator.do_final_ba && haveStates) {
      LOG(INFO) << "Final full BA...";
      estimator.doFinalBa();
    }

    // Finish up (as okvis2x_node, minus the submapping interface).
    estimator.stopThreading();
    if (haveStates) {
      estimator.setFinalTrajectoryCsvFile(
          csvPath + "/okvis2-final_trajectory.csv", false);
      estimator.writeFinalTrajectoryCsv();
    } else {
      LOG(WARNING) << "estimator never initialised — no trajectory CSV written";
    }
    g_stats.cleanExit = 1;
    writeStats();
  } catch (const std::exception &e) {
    LOG(ERROR) << e.what();
    g_stats.crashes = 1;
    writeStats();
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
