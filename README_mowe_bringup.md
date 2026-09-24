# OKVIS2-X bring-up on the Mow-e rig (OV9281 stereo + SCH16T IMU)

`okvis2x-mowe` is mow-e's fork of [OKVIS2-X](https://github.com/ethz-mrl/OKVIS2-X)
(BSD-3, base commit `38043e4`, 2026-03-17). It replaces the `okvis2-mowe` fork
of okvis2 (decision: Eemil (owner), 2026-09-24): OKVIS2-X is mow-e's single
state estimator, and RTK GNSS plus wheel odometry are to be fused inside its
Ceres graph. Branch `feat/mowe-port` carries the `okvis2-mowe` amendments
(`feat/xfeat-lighterglue-frontend` @ `baf246f`) rebased by hand onto OKVIS2-X.

GNSS ingest and the wheel-odometry factor are **not** implemented yet; see
[`docs/MOWE_TODO.md`](docs/MOWE_TODO.md).

## What changed vs. upstream OKVIS2-X

| Commit (subject) | Files | What |
|---|---|---|
| build: BRISK cross-compile fix + submodule URL | `external/CMakeLists.txt`, `external/patches/brisk/`, `.gitmodules` | BRISK picks NEON vs `-mssse3` from `CMAKE_HOST_SYSTEM_PROCESSOR`; the patched copy uses `CMAKE_SYSTEM_PROCESSOR` and is copied over the submodule file at configure time (same mechanism as the DBoW2/OpenGV patches, SHA-256 guarded). supereight2 submodule URL SSH → HTTPS. |
| build: USE_MOWE_XFEAT option + okvis_xfeat_frontend | `CMakeLists.txt`, `okvis_frontend/CMakeLists.txt`, `cmake/okvisConfig.cmake.in`, `cmake/mowe-defaults.cmake`, `okvis_xfeat_frontend/` | Opt-in XFeat/LighterGlue TensorRT library; mow-e build defaults as an initial-cache file (upstream `option()` defaults untouched). |
| frontend: XFeat/TensorRT detect+describe, float-descriptor matching, LighterGlue | `Parameters.hpp`, `ViParametersReader.cpp`, `Frame.hpp`, `Frontend.hpp/.cpp`, `ThreadedSlam.cpp` | `frontend_parameters.xfeat` switches BRISK → XFeat at runtime; cosine distance on 64-D float descriptors; LighterGlue pair proposals for stereo and top-N motion stereo; DBoW loop closure off under XFeat. |
| releasing keyframe images from memory | `ViSlamBackend.cpp`, `okvis_cv/…Frame*.hpp`, `…MultiFrame*.hpp` | Cherry-pick of okvis2 `45dade8` (missing in OKVIS2-X): frees images of pose-graph frames. |
| ros2: okvis2x_node_mowe + Subscriber QoS + camera_decimation | `okvis_ros2/src/okvis2x_node_mowe.cpp`, `okvis_ros2/src/Subscriber.cpp`, `CMakeLists.txt`, `package.xml` | Direct OV9281 ingest via `mowe_camera_core` (was `okvis_node_mowe`); best-effort QoS on the topic path. |
| ros2: okvis2x_node: guard null seInterface | `okvis_ros2/src/okvis2x_node.cpp` | Upstream bug: VI-only shutdown dereferenced a null submapping interface. |
| config: … → config/mowe/ + launch files | `config/mowe/{okvis2,okvis2-xfeat,se2}.yaml`, `okvis_ros2/launch/okvis2x_mowe_{direct,ov9281}.launch.xml` | Rig configs in the OKVIS2-X layout, new `gps_parameters` block. |

Renamed relative to `okvis2-mowe`:

| okvis2-mowe | okvis2x-mowe |
|---|---|
| `config/ov9281_sch16t.yaml` | `config/mowe/okvis2.yaml` |
| `config/ov9281_sch16t_xfeat.yaml` | `config/mowe/okvis2-xfeat.yaml` |
| — | `config/mowe/se2.yaml` (only for `okvis2x_node_subscriber`) |
| `launch/okvis_mowe_direct.launch.xml` | `okvis_ros2/launch/okvis2x_mowe_direct.launch.xml` |
| `launch/okvis_mowe_ov9281.launch.xml` | `okvis_ros2/launch/okvis2x_mowe_ov9281.launch.xml` |
| executable `okvis_node_mowe` | `okvis2x_node_mowe` |
| executable `okvis_node_subscriber` | `okvis2x_node_subscriber` (upstream rename) |

## Build flags

OKVIS2-X adds options whose upstream defaults are wrong for mow-e. They are
**not** changed in `CMakeLists.txt`; preload them with the initial-cache file
[`cmake/mowe-defaults.cmake`](cmake/mowe-defaults.cmake):

| Option | Upstream default | mow-e | Why |
|---|---|---|---|
| `USE_NN` | ON | **OFF** | No LibTorch; ON also downloads depth/FindAnything models from cvg.cit.tum.de at configure time. |
| `USE_GPU` | OFF | OFF | Torch CUDA inference, irrelevant without `USE_NN`. |
| `USE_COLIDMAP` | ON | **OFF** | Colour/object-ID (FindAnything) mapping. |
| `HAVE_LIBREALSENSE` | ON | **OFF** | No RealSense on the rig. |
| `BUILD_TESTS`, `SE_TEST`, `SE_APP` | OFF | OFF | Pinned OFF. |
| `USE_MOWE_XFEAT` | OFF (new) | ON on the Jetson | XFeat/LighterGlue frontend + `okvis2x_node_mowe`; needs `mowe_camera_core`. |
| `USE_TENSORRT` (in `okvis_xfeat_frontend`) | OFF | ON on the Jetson | Real TensorRT inference; OFF = stub (empty features). |
| `USE_SYSTEM_CERES` | OFF | optional | ON uses a system Ceres 2.2 (faster builds). |
| `BUILD_APPS`, `BUILD_ROS2` | ON | as needed | Apps need OpenCV highgui. |

Runtime (config, not CMake): `output_parameters.enable_submapping: false`
(supereight2 / PCL / GeographicLib are still compiled and linked; OKVIS2-X has
no option to remove them).

```bash
# plain CMake (e.g. a desktop check)
cmake -C cmake/mowe-defaults.cmake -S . -B build -DBUILD_ROS2=OFF
cmake --build build -j

# colcon on the Jetson / in the cross container
colcon build --packages-select mowe_camera sch16t_imu_node okvis \
  --cmake-args -C "$PWD/src/okvis2x-mowe/cmake/mowe-defaults.cmake" \
               -DUSE_MOWE_XFEAT=ON -DUSE_TENSORRT=ON \
               -Dmowe_camera_core_DIR=<prefix>/lib/cmake/mowe_camera_core
```

Submodules: `git submodule update --init --recursive` (brisk, DBoW2, OpenGV,
Ceres, googletest, supereight2 + its srl_projection). All URLs are HTTPS now.
The ROS 2 build also needs `language_feature_msgs`
(github.com/ethz-mrl/language_feature_msgs) in the workspace — OKVIS2-X
requires it unconditionally.

## Run

Direct ingest (preferred; camera owned by the node, IMU over ROS):

```bash
ros2 launch okvis okvis2x_mowe_direct.launch.xml              # XFeat config
ros2 launch okvis okvis2x_mowe_direct.launch.xml \
  config_filename:=$(ros2 pkg prefix okvis)/share/okvis/config/mowe/okvis2.yaml   # BRISK A/B
```

| Arg | Default | Notes |
|---|---|---|
| `config_filename` | `config/mowe/okvis2-xfeat.yaml` | OKVIS config |
| `camera_device` / `camera_fps` | `/dev/video1` / `50.0` | OV9281 stereo (2560×800 GREY, split by the driver) |
| `camera_decimation` | `2` | feed every Nth frame (25 Hz into the estimator) |
| `csv_path` | `/tmp/` | final trajectory CSV on shutdown |
| `imu` / `rviz` | `true` | |

Topic path (sensor nodes publish images; BRISK by default):

```bash
ros2 launch okvis okvis2x_mowe_ov9281.launch.xml
```

This path uses the stock `okvis2x_node_subscriber`, which always parses
`se_config_filename` (default `config/mowe/se2.yaml`, parsed but unused with
`enable_submapping: false`). `okvis2x_node_mowe` needs no se2 config and
refuses to start if `enable_submapping` is true.

Stop cleanly (final BA if `do_final_ba`, then the CSV) with Ctrl-C or
`ros2 service call /okvis/shutdown std_srvs/srv/SetBool "{data: true}"`.

## OKVIS2-X config differences (config/mowe/*.yaml)

- `cameras[].cam_model: pinhole` is **required**. Without it OKVIS2-X adds no
  camera at all (`numCameras() == 0`) and raises no error.
- `imu_parameters.s_a`, `output_parameters.display_topview` and
  `output_parameters.enable_submapping` are required.
- `gps_parameters:` (new): `data_type: cartesian`, `r_SA: [0,0,0]`
  (**TODO-MEASURE** antenna lever arm in the SCH16T frame),
  `yaw_error_threshold: 1.0`, `robust_gps_init: false`. The estimator registers
  a GNSS sensor, but nothing feeds it yet, so the graph is plain VIO.
- Do not put `": "` inside a trailing comment on a block-scalar line:
  `cv::FileStorage` keeps such comments inside the scalar and fails to parse.

## ⚠️ Calibration (unchanged from okvis2-mowe)

1. **Camera intrinsics** (`focal_length`, `principal_point`, `distortion`) —
   required; placeholders (`f≈1000 px`, centre, zero distortion) give wrong
   structure/scale. Checkerboard: `ros2 run camera_calibration cameracalibrator`.
2. **Stereo extrinsics** (baseline in `T_SC`) — placeholder 0.10 m; set from
   CAD, then Kalibr.
3. **Camera↔IMU extrinsics** — placeholder axis-aligned mount; Kalibr
   (`kalibr_calibrate_imu_camera`).
4. **IMU noise model** — SCH16T-K01 datasheet (Doc.No. 11624 Rev.6, Tables 4
   and 6); bias random-walk densities are estimates, refine with Allan variance.
5. **GNSS antenna lever arm** `gps_parameters.r_SA` — new, unmeasured.

## Timestamps and QoS (unchanged from okvis2-mowe)

- Camera (SOF) and IMU (SPI read) both stamp on `CLOCK_MONOTONIC`: one epoch.
  The residual sub-frame offset is `camera_parameters.image_delay` (start 0,
  tune; ≈ −exposure/2 moves the camera stamp to mid-exposure).
- The stereo pair is hardware-synced with identical stamps:
  `timestamp_tolerance: 0.003`.
- The sensor nodes publish `SensorDataQoS` (BEST_EFFORT). `Subscriber.cpp`
  subscribes best-effort (images: `rmw_qos_profile_sensor_data` via
  `image_transport::create_subscription`; IMU: `SensorDataQoS().keep_last(1000)`),
  otherwise a RELIABLE subscriber silently receives nothing. Depth and LiDAR
  subscriptions keep upstream QoS.
- GNSS stamps (future) must be time-of-validity on the same clock.

## Sanity checks

```bash
ros2 topic hz /imu/data_raw --qos-reliability best_effort
ros2 topic hz /okvis/okvis_odometry
ros2 topic echo /okvis/okvis_transform --once
```

## Known gaps

Verification state of this port (2026-09-24, x86_64 Ubuntu 24.04, GCC 13):
with `cmake -C cmake/mowe-defaults.cmake -DBUILD_ROS2=OFF -DBUILD_APPS=ON
-DUSE_SYSTEM_CERES=ON` (USE_MOWE_XFEAT=OFF) the whole non-ROS tree (brisk with
the patch applied, DBoW2, OpenGV, supereight2, all okvis libraries) compiles
and `okvis_app_synchronous` links; `Frontend.cpp` was additionally
syntax-checked with `OKVIS_USE_MOWE_XFEAT` defined; both `config/mowe` YAMLs
parse with the OKVIS2-X reader. **Not compiled:** the ROS 2 sources (`Subscriber.cpp`,
`okvis2x_node_mowe.cpp`, `okvis2x_node.cpp`), `okvis_xfeat_frontend`
(TensorRT/CUDA, `mowe_camera_core`). **Never run** on sensor data or on the
Jetson (only a headless smoke run of `okvis_app_synchronous` with
`config/mowe/okvis2.yaml` on synthetic static stereo + IMU + GNSS data).
Hand-adapted code that could not be compiled is marked
`// MOWE-PORT-REVIEW:` (`git grep MOWE-PORT-REVIEW`). See
`docs/MOWE_TODO.md` for GNSS, wheel odometry, place recognition, sysroot
dependencies and upstreamable fixes.
