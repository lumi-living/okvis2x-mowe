# okvis2x-mowe: open work items

Work that the port of `okvis2-mowe` onto OKVIS2-X deliberately did **not** do.
Decision context: Eemil (owner), 2026-09-24 — OKVIS2-X is the single state
estimator; RTK GNSS (u-blox ZED-F9P) and wheel-encoder odometry are fused
**inside** the OKVIS2-X Ceres graph (no GTSAM/iSAM2, no loosely coupled fuser).

Line numbers refer to OKVIS2-X `38043e4` plus this branch. Background and
sources: mow-e `docs/audits/okvis2x-research-2026-09-24.md` (§2 GNSS,
§3 wheel odometry, §6 build, §7 runtime).

Status legend: **NEW** = new feature, **FIX** = defect/gap, **BUILD** =
build/deploy, **VERIFY** = needs a measurement or a run.

---

## 1. GNSS (RTK) into the graph — NEW

What exists in OKVIS2-X (C++ API only, no ROS path):

| Piece | Where |
|---|---|
| Factor `okvis::ceres::GpsErrorAsynchronous` (`SizedCostFunction<3,7,9,7>`: `T_WS`, `SpeedAndBias` of the preceding state, `T_GW`), IMU-preintegrated to the GNSS time | `okvis_ceres/include/okvis/ceres/GpsErrorAsynchronous.hpp:41`, `okvis_ceres/src/GpsErrorAsynchronous.cpp` |
| `T_GW` world→GNSS-frame block, 4-DoF (`PoseManifold4d`: x, y, z, yaw), one per graph, added for every run | `ViGraph.hpp:792` (`State::T_GW`), `ViGraph.cpp:340-345` |
| Public entry points | `ThreadedSlam::addGpsMeasurement(stamp, pos, sigma)` `ThreadedSlam.cpp:350`; `ThreadedSlam::addGeodeticGpsMeasurement(stamp, lat, lon, h, hAcc, vAcc)` `ThreadedSlam.cpp:377` |
| Queue → graph | `gpsMeasurementsReceived_` drained into `gpsMeasurementDeque_` before each frame (`ThreadedSlam.cpp:627`); inserted by `estimator_.addGpsMeasurementsOnAllGraphs(...)` right after `frontend_.dataAssociationAndInitialization(...)` (`ThreadedSlam.cpp:834-845`) → `ViSlamBackend::addGpsMeasurementsOnAllGraphs` (`ViSlamBackend.cpp:58`) → `ViGraph::addGpsMeasurements` (`ViGraph.cpp:1216`) |
| Init / freeze state machine (`Off → Idle → Initialising → Initialised`, `ReInitialising`) | `ViGraph::initializationStrategy` (`ViGraph.cpp:1318`, hard-coded yaw σ < 5° at `:1331`), `checkForGpsInit`, `freezeGpsExtrinsics` (`ViGraph.cpp:835`) |
| Robust mode gates (SPP-tuned, hard-coded) | `ViGraph::checkValidGpsMeasurements` (`ViGraph.cpp:1128`); Cauchy(3.0) loss `cauchyGpsLossFunctionPtr_` (`ViGraph.cpp:236`) |
| Outputs | `State::T_GW`, `State::gpsPoints` in the optimised-graph callback (`ViInterface.hpp:186-187`); `ThreadedSlam::writeGlobalTrajectoryCsv` (`ThreadedSlam.cpp:1465`, antenna position only) |

To do:

1. **ROS 2 GNSS ingest node / subscription** (in `okvis2x_node_mowe.cpp`, marked
   `TODO(mow-e, docs/MOWE_TODO.md)`): subscribe to the `ublox_dgnss-mowe`
   output (`sensor_msgs/NavSatFix` or the HP fix), gate on RTK FIXED (inflate σ
   for FLOAT, drop worse), convert to **ENU in a fixed lawn datum** in mow-e
   code (config `gps_parameters.data_type: cartesian`, already set), and call
   `addGpsMeasurement(stamp, p_ENU, sigma)`. Do not use `geodetic`: OKVIS2-X
   resets its ENU origin to the first fix of every run
   (`globCartesianFrame_.Reset`, `ViGraph.cpp:1272`).
2. **Full covariance**: the `ThreadedSlam` API takes per-axis σ only; extend it
   (`GpsSensorReadings`, `Measurements.hpp:186`, already has a full-covariance constructor)
   if `NavSatFix.position_covariance` off-diagonals matter.
3. **Time base**: stamps must be time-of-validity on the IMU clock
   (CLOCK_MONOTONIC on the bench). The F9P gives GPS time; the ingest node must
   map it. There is no online GNSS time-offset estimation.
4. **FIX — GNSS dropped on non-keyframes.** `ViGraphEstimator::eliminateStateByImuMerge`
   (`okvis_ceres/src/ViGraphEstimator.cpp:38`) has the GNSS-factor carry-over
   commented out (`:87-115`) and erases the state from `gpsStates_` (`:166`).
   Every fix attached to a frame that is not a keyframe is lost once it leaves
   the `num_imu_frames` window — on a slow mower most fixes. Either implement
   the merge (re-anchor the factor on `refId` with the merged IMU deque) or
   force a keyframe per accepted fix.
5. **Global-frame output**: publish `T_GW`-composed pose (and a `map`→`odom`
   TF) from the optimised-graph callback; `okvis::Publisher` only publishes in
   `world` and broadcasts no TF.
6. **VERIFY** `gps_parameters.r_SA` (antenna phase centre in the SCH16T frame);
   currently `[0,0,0]` placeholder in `config/mowe/okvis2*.yaml`.
7. **VERIFY** `yaw_error_threshold: 1.0` and `robust_gps_init: false` on real
   RTK data (thresholds in robust mode are tuned for SPP: 4 m RANSAC inliers,
   3σ gates, σ_h > 6 m reject).

## 2. Wheel-encoder odometry factor — DONE (T-0125, 2026-09-27)

Built as designed below (`WheelOdometryError` = `SizedCostFunction<4,7,9>`, IMU-preintegrated
like `GpsErrorAsynchronous`; merge in `eliminateStateByImuMerge`; backlog in `ViSlamBackend`;
`ThreadedSlam::addWheelMeasurement`; `DatasetReader` `wheel0/data.csv`; `/wheel/speeds` in
`Subscriber.cpp` + `okvis2x_node_mowe`; gtest `TestWheelOdometryError.cpp`; config
`config/tumvi/okvis2-gnss-wheel.yaml`). Original plan kept for reference:

1. `okvis_ceres/include/okvis/ceres/WheelOdometryError.hpp` + `src/WheelOdometryError.cpp`:
   `WheelOdometryError` on `(T_WS, SpeedAndBias)` of the preceding state,
   IMU-preintegrated to the measurement time like `GpsErrorAsynchronous`
   (or with an `append()` like `ImuError` so it survives state elimination).
   Residual: body-frame velocity `e = [v_x^enc, 0, 0] − C_BS·(C_WSᵀ·v_W + ω_S × r_SB)`
   plus a yaw-rate term against the gyro; inflated lateral σ for skid-steer.
2. `okvis_common/include/okvis/Measurements.hpp`: `WheelSensorReadings`,
   `WheelMeasurementDeque`. `Parameters.hpp` / `ViParametersReader.cpp`:
   `wheel_parameters:` block (T_BS lever arm, σ_v, σ_lat, σ_ω, slip gates).
3. `ViGraph.hpp/.cpp`: `std::vector<WheelFactor>` in `State` (next to
   `GpsFactors`, `ViGraph.hpp:806`), `addWheelMeasurement()`.
   `ViGraphEstimator::eliminateStateByImuMerge` must **merge** wheel factors
   into the previous state (the GNSS precedent drops them, see 1.4).
4. `ViSlamBackend`: `addWheelMeasurementsOnAllGraphs()` with a backlog while
   loop closure runs (like `addGpsBacklog_`); both `realtimeGraph_` and
   `fullGraph_`.
5. `ThreadedSlam`: `threadsafe::Queue` + `addWheelMeasurement()`, inserted next
   to `addGpsMeasurementsOnAllGraphs` (`ThreadedSlam.cpp:845`).
6. ROS: subscribe to the CAN-bridge odometry topic in `okvis2x_node_mowe`.
7. Slip gating: encoder yaw rate vs gyro, encoder speed vs `v_W` projected to
   B; skip or inflate on large innovations (template:
   `ViGraph::checkValidGpsMeasurements`), plus a Cauchy/Huber loss.
   Generic fallback hook: `ViGraph::addRelativePoseConstraint`
   (`ViGraph.cpp:787`, `RelativePoseError<6,7,7>`), but relative-pose links are
   removed on eliminated states, so it only works between keyframes.

## 3. Build and deployment

- **BUILD — cross toolchain**: `tools/cross` (only on rig-ubuntu, not on mow-e
  `main`) must fetch this fork instead of okvis2 `a2ea006`, and its in-place
  BRISK `sed` (`patch-sources.sh`) becomes unnecessary: the fix is now
  `external/patches/brisk/` + the configure-time copy in `external/CMakeLists.txt`.
- **BUILD — sysroot deps**: OKVIS2-X hard-requires GeographicLib, PCL and
  supereight2 (OpenMP) in the core libraries; all must be in the Jetson
  sysroot. `CMakeLists.txt:42` hard-codes
  `list(APPEND CMAKE_MODULE_PATH "/usr/share/cmake/geographiclib")`; for a
  sysroot build it should become `${CMAKE_SYSROOT}/usr/share/cmake/geographiclib`
  (not changed here: untested).
- **BUILD — `language_feature_msgs`** is `find_package(... REQUIRED)` for every
  `BUILD_ROS2` build (`CMakeLists.txt`, ROS section) although only the
  `USE_NN`/`USE_COLIDMAP` network nodes use it: vendor
  github.com/ethz-mrl/language_feature_msgs into the workspace, or gate the
  `find_package` / `ament_target_dependencies` entry on `USE_COLIDMAP`.
- **BUILD — `mowe_camera_core` ordering**: `okvis2x_node_mowe` needs
  `mowe_camera_core` installed first; it is not a `package.xml` dependency
  (optional, `USE_MOWE_XFEAT` only). Build it first or pass
  `-Dmowe_camera_core_DIR`.
- **BUILD — ROS 2 distro**: `image_transport::create_subscription(node*, …,
  rmw_qos_profile_t)` is the Humble API used in `Subscriber.cpp`; recheck on
  Jazzy (deprecations) if the robot moves to 24.04.
- **VERIFY — nothing ROS-side was compiled** in the porting environment
  (`Subscriber.cpp`, `okvis2x_node_mowe.cpp`, `okvis2x_node.cpp`), nor the
  TensorRT/CUDA sources. First colcon build on the Jetson is the real test;
  the hand-adapted spots are marked `// MOWE-PORT-REVIEW:`.

## 4. Frontend / place recognition

- **FIX — 48-byte assumptions left**: `ViSlamBackend::saveMap` (`ViSlamBackend.cpp`, `for(size_t i=0; i<48; ++i)`)
  and `Component::save/load` (`Component.cpp`) still read/write 48 descriptor
  bytes; with XFeat (256 B) saved maps are wrong. `loadComponent` asserts
  under XFeat; `saveMap` does not. Same as in okvis2-mowe.
- **NEW — place recognition**: DBoW2 (`FBrisk`, BRISK vocabulary) is off under
  XFeat; DINOv2/AnyLoc + FAISS replacement at `Frontend::getFilteredDBoWResult`
  / the `DBoW` struct (ADR-0040 issue 5). DBoW2's license clause 3 (notify the
  author on binary redistribution) is another reason to replace it.
- **VERIFY — performance**: OKVIS2-X runs `removeOutliers<>` twice after
  matching (upstream behaviour, `Frontend.cpp` after `matchStereo`); measure on
  the Orin Nano before tuning `realtime_time_limit`. OKVIS2-X paper VI memory
  is 4.5–4.9 GB on a desktop; the `45dade8` cherry-pick (keyframe image
  release) is in this branch but its effect on the 8 GB Jetson is unmeasured.

## 5. Upstreamable

- `okvis2x_node.cpp` null `seInterface` guard (commit "ros2: okvis2x_node:
  guard null seInterface …") → PR to ethz-mrl/OKVIS2-X.
- BRISK `CMAKE_SYSTEM_PROCESSOR` → PR to smartroboticslab/brisk.
- `45dade8` (keyframe image release) → OKVIS2-X is missing it; PR candidate.
