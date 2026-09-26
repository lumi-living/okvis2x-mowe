/**
 * OKVIS2-X - Open Keyframe-based Visual-Inertial SLAM Configurable with Dense 
 * Depth or LiDAR, and GNSS
 *
 * Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 * Copyright (c) 2020, Smart Robotics Lab / Imperial College London
 * Copyright (c) 2025, Mobile Robotics Lab / Technical University of Munich 
 * and ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

/**
 * @file okvis_app_synchronous.cpp
 * @brief This file processes a dataset.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <memory>
#include <functional>
#include <string>
#include <vector>

#include <Eigen/Core>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#pragma GCC diagnostic pop
#include <okvis/ViParametersReader.hpp>
#include <okvis/ThreadedSlam.hpp>
#include <okvis/DatasetReader.hpp>
#include <okvis/RpgDatasetReader.hpp>
#include <okvis/TrajectoryOutput.hpp>
#include <boost/filesystem.hpp>

#include <execinfo.h>
#include <mutex>
#include <set>


/// \brief Main
/// \param argc argc.
/// \param argv argv.
int main(int argc, char **argv)
{

  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = 0;  // INFO: 0, WARNING: 1, ERROR: 2, FATAL: 3
  FLAGS_colorlogtostderr = 1;
  FLAGS_minloglevel = 0;

  // mowe (T-0120): `--preload-map <keyframes.bin>` loads another run's VPR
  // keyframe database as a foreign map (verified + counted only). Stripped
  // from argv before the positional parsing below.
  std::string preloadMap;
  // mowe (T-0124): `--states-log <file.jsonl>` records what the ROS wrapper publishes
  // (okvis2x_node_mowe): every IMU-propagated body pose ("prop") and every
  // optimised-graph callback with T_GW + gpsStatus ("opt"), so
  // mowe_localization_outputs can be replayed on the device without the estimator
  // (replay_okvis_states). Format: docs in onboard/localization/mowe_localization_outputs/README.md.
  std::string statesLog;
  {
    std::vector<char*> args;
    for (int i = 0; i < argc; ++i) {
      if (std::string(argv[i]) == "--preload-map" && i + 1 < argc) {
        preloadMap = argv[++i];
      } else if (std::string(argv[i]) == "--states-log" && i + 1 < argc) {
        statesLog = argv[++i];
      } else {
        args.push_back(argv[i]);
      }
    }
    argc = int(args.size());
    for (int i = 0; i < argc; ++i) argv[i] = args[size_t(i)];
  }

  // mowe: argc == 3 (config + dataset only) is accepted and saves to the
  // current directory, so the overnight verify can `cd out/<ticket> && run` (T-0107).
  if (argc < 3 || argc > 5) {
    LOG(ERROR)<<
    "Usage: ./" << argv[0] << " configuration-yaml-file dataset-folder [save-folder] [-rpg] [--preload-map keyframes.bin]";
    return EXIT_FAILURE;
  }

  okvis::Duration deltaT(0.0);
  bool rpg = false;
  std::string savePath = ".";
  if (argc == 5) {
    savePath = std::string(argv[3]);
    if(strcmp(argv[4], "-rpg")==0) {
      rpg = true;
    }
  }
  else if (argc == 4) {
    if(strcmp(argv[3], "-rpg")!=0)  {
      savePath = std::string(argv[3]);
    }
  }

  // read configuration file
  std::string configFilename(argv[1]);

  okvis::ViParametersReader viParametersReader(configFilename);
  okvis::ViParameters parameters;
  viParametersReader.getParameters(parameters);

  // dataset reader
  // the folder path
  std::string path(argv[2]);
  std::shared_ptr<okvis::DatasetReaderBase> datasetReader;
  if(rpg){
    datasetReader.reset(new okvis::RpgDatasetReader(
                          path, deltaT, int(parameters.nCameraSystem.numCameras())));
  } else {
    datasetReader.reset(new okvis::DatasetReader(
                          path, int(parameters.nCameraSystem.numCameras()),
                          parameters.camera.sync_cameras, deltaT, parameters.gps,
                          parameters.wheel)); // mow-e (T-0125): mav0/wheel0/data.csv
  }

  // also check DBoW2 vocabulary
  // mowe: resolve the real binary location (argv[0] is bare when found via PATH)
  // and fall back to the colcon install layout <prefix>/share/okvis/resources (T-0107).
  boost::filesystem::path executable = boost::filesystem::canonical("/proc/self/exe");
  std::string dBowVocDir = executable.remove_filename().string();
  if(!std::ifstream(dBowVocDir+"/small_voc.yml.gz").good()) {
    dBowVocDir = dBowVocDir + "/../share/okvis/resources";
  }
  std::ifstream infile(dBowVocDir+"/small_voc.yml.gz");
  if(!infile.good()) {
     LOG(ERROR)<<"DBoW2 vocabulary " << dBowVocDir << "/small_voc.yml.gz not found.";
     return EXIT_FAILURE;
  }

  okvis::ThreadedSlam estimator(parameters, dBowVocDir);
  estimator.setBlocking(true);
  if (!preloadMap.empty()) {
    // T-0120: foreign map = retrieval database of another run (false-loop test).
    if (!estimator.frontend().loadForeignKeyframes(preloadMap, parameters.nCameraSystem)) {
      LOG(ERROR) << "cannot load --preload-map " << preloadMap;
      return EXIT_FAILURE;
    }
  }

  // write logs
  std::string mode = "slam";
  if(!parameters.estimator.do_loop_closures) {
    mode = "vio";
  }
  if(parameters.camera.online_calibration.do_extrinsics) {
    mode = mode+"-calib";
  }

  const bool isWriteRpg = false;
  okvis::TrajectoryOutput writer(savePath+"/okvis2-" + mode + "_trajectory.csv", isWriteRpg, parameters.output.display_topview);
  // mowe (T-0124): JSONL states log, same content as the /okvis_odometry +
  // /okvis2x/estimator_state topics of okvis2x_node_mowe (ADR-0042 §(4)). The
  // propagation mirrors okvis::Publisher::realtimePredictAndPublish (okvis::Trajectory fed
  // with every IMU sample, updated at each callback); poses are T_WB via the static T_BS.
  struct StatesLog {
    std::ofstream f;
    std::mutex m;
    okvis::Trajectory trajectory;
    okvis::kinematics::Transformation T_SB;
    okvis::Time lastGps;  // latest state time that carries a gated GNSS factor
    static std::string pose(const okvis::kinematics::Transformation& T) {
      char b[256];
      snprintf(b, sizeof b, "[%.6f,%.6f,%.6f,%.9f,%.9f,%.9f,%.9f]", T.r()[0], T.r()[1], T.r()[2],
               T.q().x(), T.q().y(), T.q().z(), T.q().w());
      return b;
    }
    void imu(const okvis::Time& t, const Eigen::Vector3d& acc, const Eigen::Vector3d& gyr) {
      std::lock_guard<std::mutex> l(m);
      okvis::State s;
      if (!trajectory.addImuMeasurement(t, acc, gyr, s)) return;
      f << "{\"type\":\"prop\",\"t_ns\":" << uint64_t(t.toNSec()) << ",\"T_WB\":" << pose(s.T_WS * T_SB)
        << ",\"v_W\":[" << s.v_W[0] << "," << s.v_W[1] << "," << s.v_W[2] << "]}\n";
    }
    void opt(const okvis::State& state, const okvis::TrackingState& ts,
             std::shared_ptr<const okvis::AlignedMap<okvis::StateId, okvis::State>> updated,
             int gpsStatus) {
      std::lock_guard<std::mutex> l(m);
      std::set<okvis::StateId> affected;
      trajectory.update(ts, updated, affected);
      for (const auto& u : *updated)
        if (!u.second.gpsPoints.empty() && u.second.timestamp > lastGps) lastGps = u.second.timestamp;
      f << "{\"type\":\"opt\",\"t_ns\":" << uint64_t(state.timestamp.toNSec()) << ",\"id\":" << state.id.value()
        << ",\"T_WB\":" << pose(state.T_WS * T_SB) << ",\"T_GW\":" << pose(state.T_GW)
        << ",\"status\":" << gpsStatus << ",\"quality\":" << int(ts.trackingQuality)
        << ",\"keyframe\":" << (ts.isKeyframe ? 1 : 0) << ",\"loop\":" << (ts.recognisedPlace ? 1 : 0)
        << ",\"n_updated\":" << updated->size() << ",\"t_last_gps_ns\":" << uint64_t(lastGps.toNSec()) << "}\n";
    }
  } statesLogWriter;
  if (!statesLog.empty()) {
    statesLogWriter.f.open(statesLog);
    if (!statesLogWriter.f) { LOG(ERROR) << "cannot open --states-log " << statesLog; return EXIT_FAILURE; }
    statesLogWriter.T_SB = parameters.imu.T_BS.inverse();
    LOG(INFO) << "writing states log to " << statesLog;
  }
  estimator.setOptimisedGraphCallback(
        [&](const okvis::State& state, const okvis::TrackingState& ts,
            std::shared_ptr<const okvis::AlignedMap<okvis::StateId, okvis::State>> updatedStates,
            std::shared_ptr<const okvis::MapPointVector> landmarks) {
          writer.processState(state, ts, updatedStates, landmarks);
          if (statesLogWriter.f.is_open()) statesLogWriter.opt(state, ts, updatedStates, estimator.gpsAlignmentStatus());
        });
  estimator.setFinalTrajectoryCsvFile(savePath+"/okvis2-" + mode + "-final_trajectory.csv", isWriteRpg);
  estimator.setMapCsvFile(savePath+"/okvis2-" + mode + "-final_map.csv");

  // connect reader to estimator
  datasetReader->setImuCallback(
        [&](const okvis::Time& t, const Eigen::Vector3d& acc, const Eigen::Vector3d& gyr) {
          const bool ok = estimator.addImuMeasurement(t, acc, gyr);
          if (statesLogWriter.f.is_open()) statesLogWriter.imu(t, acc, gyr);
          return ok;
        });
  datasetReader->setImagesCallback(
        std::bind(&okvis::ThreadedSlam::addImages, &estimator, std::placeholders::_1,
                  std::placeholders::_2, std::placeholders::_3));
  if(parameters.gps) {
    if ((*parameters.gps).type == "cartesian") {
      datasetReader->setGpsCallback(
              std::bind(&okvis::ThreadedSlam::addGpsMeasurement, &estimator,
                        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    } else if ((*parameters.gps).type == "geodetic" || (*parameters.gps).type == "geodetic-leica") {
      datasetReader->setGeodeticGpsCallback(
              std::bind(&okvis::ThreadedSlam::addGeodeticGpsMeasurement, &estimator,
                        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                        std::placeholders::_4, std::placeholders::_5, std::placeholders::_6));
    } else {
      LOG(ERROR) << "Unknown GPS data type.";
      return EXIT_FAILURE;
    }
  }
  if(parameters.wheel) { // mow-e (T-0125, ADR-0042 design item 3)
    datasetReader->setWheelCallback(
            std::bind(&okvis::ThreadedSlam::addWheelMeasurement, &estimator,
                      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                      std::placeholders::_4, std::placeholders::_5));
  }

  // start
  okvis::Time startTime = okvis::Time::now();
  datasetReader->startStreaming();
  int progress = 0;
  while (true) {
    estimator.processFrame();
    std::map<std::string, cv::Mat> images;
    estimator.display(images);
    for(const auto & image : images) {
      cv::imshow(image.first, image.second);
    }
    cv::Mat topView;
    writer.drawTopView(topView);
    if(!topView.empty()) {
      cv::imshow("OKVIS 2 Top View", topView);
    }
    if(!images.empty() || !topView.empty()) {
      char b = cv::waitKey(2);
      if (b == 's') {
        cv::imwrite("saved.png", topView);
      }
    }

    // check if done
    if(!datasetReader->isStreaming()) {
      estimator.stopThreading();
      LOG(INFO) << "Finished!" << std::endl;
      estimator.writeFinalTrajectoryCsv();
      if(parameters.gps){
        estimator.writeGlobalTrajectoryCsv(savePath+"/okvis2-" + mode + "-global-final_trajectory.csv");
        // mow-e (T-0117): IMU origin in G (what an IMU-pose ground truth compares to)
        // and the GNSS bookkeeping next to the trajectories.
        estimator.writeGlobalTrajectoryCsv(savePath+"/okvis2-" + mode + "-global_trajectory.csv", false);
        estimator.writeGnssStatsJson(savePath+"/gnss_stats.json");
      }
      if(parameters.wheel){ // mow-e (T-0125)
        estimator.writeWheelStatsJson(savePath+"/wheel_stats.json");
      }
      estimator.setFinalTrajectoryCsvFile(savePath+"/okvis2-" + mode + "-final-ba_trajectory.csv", isWriteRpg);
      if(parameters.estimator.do_final_ba) {
        LOG(INFO) << "final full BA...";
        cv::Mat topView;
        estimator.doFinalBa();
        writer.drawTopView(topView);
        if (!topView.empty()) {
          cv::imshow("OKVIS 2 Top View Final", topView);
          cv::imwrite("okvis2_final_ba.png", topView);
        }
        cv::waitKey(1000);
      }
      estimator.writeFinalTrajectoryCsv();
      if(parameters.gps){
        estimator.writeGlobalTrajectoryCsv(savePath+"/okvis2-" + mode + "-global-final-ba_trajectory.csv");
      }
      if(parameters.estimator.do_final_ba) {
        estimator.saveMap();
      }
      LOG(INFO) <<"total processing time " << (okvis::Time::now() - startTime) << " s" << std::endl;
      // mowe: front-end statistics next to the trajectories (T-0113).
      estimator.frontend().writeStatsJson(savePath+"/frontend_stats.json");
      // mowe (T-0120): VPR loop-closure funnel + this run's keyframe database.
      if (estimator.frontend().usingVprLoopClosure()) {
        estimator.frontend().writeLoopStatsJson(savePath+"/loop_stats.json");
        estimator.saveKeyframes(savePath+"/keyframes.bin");
      }
      break;
    }

    // display progress
    int newProgress = int(datasetReader->completion()*100.0);
#ifndef DEACTIVATE_TIMERS
    if (newProgress>progress) {
      LOG(INFO) << okvis::timing::Timing::print();
    }
#endif
    if (newProgress>progress) {
      progress = newProgress;
      LOG(INFO) << "Progress: " << progress << "% ";
    }
  }
  return EXIT_SUCCESS;
}