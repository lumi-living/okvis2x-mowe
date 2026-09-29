<div align="center">
<h1>OKVIS2-X: Open Keyframe-based Visual-Inertial SLAM Configurable with Dense Depth or LiDAR, and GNSS</h1>
<a href="https://ieeexplore.ieee.org/abstract/document/11196039?casa_token=CCIXQf90ensAAAAA:0lXDqeqgngNWTe0dEMIxEji3vIF2W97xq0vSeLxLBEfmC3YfDb6PSiFTQuxZ9NMGkh2saUTF"><img src="https://img.shields.io/badge/TRO-Paper-brightgreen?style=flat"></a>
<a href="https://arxiv.org/abs/2510.04612"><img src="https://img.shields.io/badge/arXiv-2510.04612-blue?style=flat"></a>
<a href="https://www.youtube.com/watch?v=K8oZvbI7I58"><img src="https://img.shields.io/badge/YouTube-red?style=flat"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-BSD3-yellow?style=flat"></a>
<br>
<strong>
<a href="https://srl.cit.tum.de/members/boche">Simon Boche<sup>*1</sup></a> &nbsp;&nbsp;
<a href="https://lastflowers.github.io">Jaehyung Jung<sup>*1</sup></a> &nbsp;&nbsp;
<a href="https://srl.cit.tum.de/members/barba">Sebastián Barbas Laina<sup>*1</sup></a> &nbsp;&nbsp;
<a href="https://mrl.ethz.ch/the-group/people/lab-members/stefan-leutenegger.html">Stefan Leutenegger<sup>2</sup></a>
<br>
<sup>*</sup> Equal contribution &nbsp;&nbsp;
<sup>1</sup> Technical University of Munich &nbsp;&nbsp;
<sup>2</sup> ETH Zurich
</strong>
</div>

<div align="center">
<h1>FindAnything: Open-Vocabulary and Object-Centric Mapping for Robot Exploration in Any Environment</h1>
<a href="https://arxiv.org/abs/2504.08603"><img src="https://img.shields.io/badge/arXiv-2504.08603-blue?style=flat"></a>
<a href="https://www.youtube.com/watch?v=ovmWyqGhA1o"><img src="https://img.shields.io/badge/YouTube-red?style=flat"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-BSD3-yellow?style=flat"></a>
<br>
<strong>
<a href="https://srl.cit.tum.de/members/barba">Sebastián Barbas Laina<sup>*1</sup></a> &nbsp;&nbsp;
<a href="https://srl.cit.tum.de/members/boche">Simon Boche<sup>*1</sup></a> &nbsp;&nbsp;
<a href="https://sotiris.papatheodorou.xyz/">Sotiris Papatheodorou<sup>*2</sup></a> &nbsp;&nbsp;
<a href="https://srl.cit.tum.de/members/schaefer">Simon Schaefer<sup>1</sup></a> &nbsp;&nbsp;
<a href="https://lastflowers.github.io">Jaehyung Jung<sup>1</sup></a> &nbsp;&nbsp;
<a href="https://helenol.com/research/">Helen Oleynikova<sup>3</sup></a> &nbsp;&nbsp;
<a href="https://mrl.ethz.ch/the-group/people/lab-members/stefan-leutenegger.html">Stefan Leutenegger<sup>3</sup></a>
<br>
<sup>*</sup> Equal contribution &nbsp;&nbsp;
<sup>1</sup> Technical University of Munich &nbsp;&nbsp;
<sup>2</sup> University of Patras &nbsp;&nbsp;
<sup>3</sup> ETH Zurich
</strong>
</div>

### Overview (OKVIS2-X)
OKVIS2-X is a multi-sensor SLAM system based on a factor graph, and is a non-trivial extension of the sparse, landmark-based [OKVIS2](https://github.com/ethz-mrl/okvis2). OKVIS2-X supports fusing multiple cameras and an IMU, with optional integration of a GNSS receiver and LiDAR or depth images (from a depth network or depth sensor). For map representation, OKVIS2-X uses submap-based volumetric occupancy to seamlessly support robot exploration and navigation.

<div align="center">
  <img src="resources/readme/okvis2x-showcase.gif" alt="Demo">
</div>

### Overview (FindAnything)
FindAnything is an open-world mapping framework that incorporates vision-language information into dense volumetric submaps. Thanks to the use of vision-language features, FindAnything combines pure geometric and open-vocabulary semantic information for a higher level of understanding. It proposes an efficient storage of open-vocabulary information through the aggregation of features at the object level. Pixelwise vision-language features are aggregated based on eSAM segments, which are in turn integrated into object-centric volumetric submaps, providing a mapping from open-vocabulary queries to 3D geometry that is scalable also in terms of memory usage.

<div align="center">
  <a href="https://www.youtube.com/watch?v=ovmWyqGhA1o">
    <img src="resources/readme/findanything-topfigure.png" alt="FindAnything Demo">
  </a>
</div>

---

### Related Papers
<details>
  <summary>Click to expand</summary>

  [1] [OKVIS2-X: Open Keyframe-based Visual-Inertial SLAM Configurable with Dense Depth or LiDAR, and GNSS (IEEE T-RO 2025 - Special Issue Visual SLAM)](https://arxiv.org/abs/2510.04612)

  [2] [FindAnything: Open-Vocabulary and Object-Centric Mapping for Robot Exploration in Any Environment (ICRA 2026)](https://arxiv.org/abs/2504.08603) 
  
  [3] [OKVIS2: Realtime Scalable Visual-Inertial SLAM with Loop Closure (arXiv 2022)](https://arxiv.org/abs/2202.09199)

  [4] [Tightly-coupled LiDAR-visual-inertial SLAM and large-scale volumetric occupancy mapping (ICRA 2024)](https://arxiv.org/abs/2403.02280)

  [5] [Uncertainty-Aware Visual-Inertial SLAM with Volumetric Occupancy Mapping (ICRA 2025)](https://arxiv.org/abs/2409.12051)

  [6] [Visual-Inertial SLAM with Tightly-Coupled Dropout-Tolerant GPS Fusion (IROS 2022)](https://arxiv.org/abs/2208.00709)

  [7] [Multi-Resolution 3D Mapping with Explicit Free Space Representation for Fast and Accurate Mobile Robot Motion Planning (RA-L 2021)](https://arxiv.org/abs/2010.07929)

  This is the implementation of [1]&[2]. It is further based on the work presented in [3-7].

  If you publish work that relates to this software, please cite at least [1] or [2].
</details>

```tex
@article{boche2025okvis2x,
  author={Boche, Simon and Jung, Jaehyung and Laina, Sebastián Barbas and Leutenegger, Stefan},
  journal={IEEE Transactions on Robotics}, 
  title={OKVIS2-X: Open Keyframe-Based Visual-Inertial SLAM Configurable With Dense Depth or LiDAR, and GNSS}, 
  year={2025},
  volume={41},
  number={},
  pages={6064-6083},
  doi={10.1109/TRO.2025.3619051}}

```

```tex
@article{laina2025findanything,
  title={FindAnything: Open-Vocabulary and Object-Centric Mapping for Robot Exploration in Any Environment},
  author={Laina, Sebasti{\'a}n Barbas and Boche, Simon and Papatheodorou, Sotiris and Schaefer, Simon and Jung, Jaehyung and Oleynikova, Helen and Leutenegger, Stefan},
  journal={arXiv preprint arXiv:2504.08603},
  year={2025}
}
```
---

### License ###

The 3-clause BSD license (see file LICENSE) applies.

--- 

### Overview
1. [Setup](#how-do-i-get-set-up)
2. [Running the apps](#running-the-synchronous-applications-for-dataset-processing)
3. [Supported Dataset Formats](#dataset-format)
4. [Configuration](#configuration)
5. [Output](#outputs-and-frames)
6. [ROS2](#building-the-project-with-ros2)
7. [Contact](#contribution-guidelines)

---

### How do I get set up? ###
#### Ubuntu ####

OKVIS2-X and FindAnything have both been tested on Ubuntu 20.04, 22.04, and 24.04.

This is a pure cmake project.
You will need to install the following dependencies,

* CMake,

        sudo apt install cmake

  In Ubuntu 20.04, please upgrade your cmake &ge; 3.17 or checkout to v2.1.0 in `external/ceres-solver`.

* google-glog + gflags,

        sudo apt install libgoogle-glog-dev

* BLAS & LAPACK,

        sudo apt install libatlas-base-dev

* Eigen3,

        sudo apt install libeigen3-dev

* SuiteSparse and CXSparse,

        sudo apt install libsuitesparse-dev

* Boost,

        sudo apt install libboost-dev libboost-filesystem-dev

* OpenCV 2.4-4: follow the instructions on http://opencv.org/ or install
  via

        sudo apt install libopencv-dev

* GeographicLIB in **Ubuntu 20 and Ubuntu 22**

        sudo apt-get install libgeographic-dev
    
* GeographicLIB in **Ubuntu 24**

        sudo apt-get install libgeographiclib-dev

* PCL

        sudo apt-get install libpcl-dev
  
* LibTorch: If you want to run **FindAnything** or if you want to run depth networks (stereo network and multi-view stereo network) or segmentation CNN to remove Sky points etc, install with instructions from the link below. Get the C++ version with C++11 ABI with or without CUDA
  (depending on availability on your machine):

    https://pytorch.org/get-started/locally/

    Also, depending on where you downloaded it to, you may want to tell cmake in your `~.bashrc`:

        export Torch_DIR=/path/to/libtorch

    Furthermore, you can turn on the NVIDIA GPU to be used for inference, if you have one, with
    `USE_GPU=ON`.

    In case you absolutely do not want to use `LibTorch`, you may disable with `USE_NN=OFF`.

* Optional: you can use this package with a Realsense D435i or D455.
Follow the instructions on:
https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md

* Optional: if you need to use openBLAS library, please use the openMP version of it:

        sudo apt-get install libopenblas-openmp-dev

* Optional: you can use this package as part of a ROS2 workspace. In this case set `BUILD_ROS2=ON`.
  Assuming you are running Ubuntu 24.04, follow the steps here:
  https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debians.html

### Building the project (no ROS2) ###

Clone the OKVIS2-X repository:

    git clone --recurse-submodules git@github.com:ethz-mrl/OKVIS2-X.git

If you forgot the `--recurse-submodules` flag when cloning run the following
command in the repository root:

    git submodule update --init --recursive

To change the cmake build type for the whole project use:

    mkdir build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j<num_job>

When running the command `cmake -DCMAKE_BUILD_TYPE=Release ..` keep in mind that based on your optional configurations, you might have to add the following command line options. Nonetheless, if you want to run **FindAnything**, make sure to set to ON the the color and id mapping `-DUSE_COLIDMAP=ON`

If you do not have a realsense installed:

    cmake -DCMAKE_BUILD_TYPE=Release -DHAVE_LIBREALSENSE=OFF ..

If you do not have Torch: 

    cmake -DCMAKE_BUILD_TYPE=Release -DUSE_NN=OFF ..

If you have none of the above:

    cmake -DCMAKE_BUILD_TYPE=Release -DUSE_NN=OFF -DHAVE_LIBREALSENSE=OFF ..

### Running the synchronous applications for dataset processing ###
To run a synchronous processing of datasets (as used for the paper results), there are the following apps:
* `okvis_app_synchronous`: Visual(-Inertial) Mode with optional Fusion of GNSS
* `okvis2x_app_synchronous`: Runs OKVIS2-X with dense mapping either based on provided depth images or LiDAR measurements
* `okvis2x_app_snetwork_synchronous`: Runs OKVIS2-X with dense mapping based on Depth Prediction from the Stereo Network
* `okvis2x_app_depthfusion_synchronous`: Runs OKVIS2-X with dense mapping based on Depth Fusion (adding MVS network)
* `okvis2x_app_language_synchronous`: FindAnything (see [here](eval/findanything/replica/) for evaluation details)

The apps can be run as follows:
```bash
./okvis2x_app_[*]_synchronous [okvis2-config] [se2-config] /path/to/dataset/ /path/to/output/directory/

# Example 1) Visual-Inertial-LiDAR configuration
./okvis2x_app_synchronous ../config/vbr/okvis2-lidar-driving.yaml ../config/vbr/se2-lidar-driving.yaml /path/to/vbr/campus0/ /path/to/output/directory/ 

# Example 2) Visual-Inetial-StereoDepth configuration
./okvis2x_app_snework_synchronous ../config/euroc/okvis2.yaml ../config/euroc/se2.yaml /path/to/euroc/mav0/ /path/to/output/directory/

# Example 3) Visual-Inetial-DepthFusion configuration
./okvis2x_app_depthfusion_synchronous ../config/euroc/okvis2.yaml ../config/euroc/se2.yaml /path/to/euroc/mav0/ /path/to/output/directory/

# Example 4) FindAnything
./okvis2x_app_language_synchronous ../config/replica/okvis2.yaml ../config/replica/se2.yaml /path/to/replica/dataset/ /path/to/output/directory/
```
* `okvis2-config`: parameter file for state-estimation related parameters
* `se2-config`: parameter file for mapping related parameters

You find examples for the datasets used in the paper in the respective subfolders in `./config`.

#### mow-e: deterministic blocking replay (T-0127, T-0131)
Replay determinism is a test-infrastructure property of **blocking** mode (`setBlocking(true)`, i.e. the
synchronous apps). Live mode keeps its threads, time limits and asynchronous loop-closure import unchanged.
In blocking mode two replays of one binary on one dataset give bit-identical trajectories because:
* the realtime solve and the pose-only solves run with 1 thread, and OpenGV RANSAC has a fixed seed (T-0127);
* `estimator_parameters.full_graph_join_frames: N` (N >= 0; default -1 = live behaviour) runs the
  loop-closure full-graph solve with 1 thread, starts it before the frame's realtime solve, joins it N frames
  later on the main thread and imports it in exactly that frame (`config/tumvi/okvis2.yaml` uses 8, which reproduces the live import lag: room1 84 loop closures vs 79-85 live; 0 gives 107);
* Ceres' fast parameter-block removal is patched (`external/CMakeLists.txt`) to remove residuals in program
  order, not heap-address order, since the residual order drives the floating-point summation order;
* `matchToMap` no longer matches landmarks whose observations were all pruned against an uninitialised
  descriptor row.

`MOWE_DET_TRACE=<file>` writes per-frame stage hashes (D detection, P estimator before association,
A association, O optimised, M marginalised, F full-graph import); diff two sorted traces to find the first
divergent stage. `MOWE_DET_DUMP_A=1` adds per-keypoint landmark ids.

### Output Files
<details>
  <summary>Click to expand</summary>
  In general, the apps will produce the following output files:


  * Estiamted Trajectories as `*.csv` files:
    * `okvis2-[slam/vio]_trajectory.csv` : causal trajectory
    * `okvis2-[slam/vio]-final_trajectory.csv` : non-causal (loop-closed) trajectory
    * `okvis2-[slam/vio]-final-ba_trajectory.csv` : after final Bundle Adjustment (BA, if specified in config file)
  * (with dense mapping enabled): Meshes of submaps (*.ply format); can be viewed e.g. in ParaView or MeshLab
  * Additional FindAnything outputs:
    * Mesh Faces colored according to Object ID
    * `objectMap[submapId].csv`: Export of Object/Segment Informatio (ID, #vertices, centroid, bounding box, feature vector)
    * `activationMesh_[keyframeId].ply`: Mesh with coloring according to feature similarity (right now, the feature is hard-coded in the Non-ROS setup)

  The result directory can be specified via the command line when running the application.
</details>

### Dataset Format ###
<details>
  <summary>Click to expand</summary>

  The demo examples assume datasets in an extended [Euroc format](https://ethz-asl.github.io/datasets/euroc-mav/) (ASL/ETH format).

  ```md
  [dataset_folder]
  ├── cam0
  │   ├── data
  │   │   ├── *.png
  │   └── data.csv
  ├── ...
  ├── camn
  │   ├── data
  │   │   ├── *.png
  │   └── data.csv
  ├── imu0
  │   └── data.csv
  ├── depth0
  │   ├── data
  │   │   ├── *.tiff
  │   └── data.csv
  ├── lidar0
  │   └── data.csv
  └── gps0
     └── [data.csv | data-raw.csv]
  ```
  where camera and imu follow the standard EuRoC format.

  For the `depth0` subfoler, it has the same format as in `cam`. Our framework assumes metric depth. If you are using png for the depth images, scale should be accounted in the reader. For example, you need a scale facor of 5000 (TUM format) when reading the depth images. This can be specified in the `se2-config` with the following parameter, else it will be assumed that depth is metric.
  ```yaml
  general:
    depth_scaling_factor:       0.0002
  ```

  LiDAR assumes the LiDAR points to be stored in a csv file with the following format (Intensity currently not used)
  ```
  timestamp [ns], x, y, z, Intensity
  ```
  You find scripts to convert bag files (as from the HILTI22 SLAM challenge: https://hilti-challenge.com/dataset-2022.html or the VBR dataset: https://rvp-group.net/slam-dataset.html) in `tools/[hilti/vbr]_bag2mrl.py` which will extract the required data from the bag file. It is used as follows:
  ```bash
  python3 [hilti/bag]_bag2mrl.py [BAG1.bag BAG2.bag ... BAGN.bag]
  
  # will create folders BAG1 ... BAGN with the required data format
  ```

  For fusion of GPS / global position measurements, the following formats are supported:
  * `cartesian`: file `[dataset_path]/gps0/data.csv`
  ```
  timestamp, x, y, z, hErr1, hErr2, vErr
  ```
  * `geodetic`: file `[dataset_path]/gps0/data_raw.csv`
  ```
  timestamp, latitude, longitude, altitude, horizontal_error, vertical_error, n_sat
  ```
  * `geodetic-leica`: file `[dataset_path]/gnss.csv` (a lot of the fields are not used)
  ```csv
  TS,DATE,TIME,FIX,RTK,NUM_SV,LAT,LON,HEIGHT,HMSL,H_ACC,V_ACC,VEL_N,VEL_E,VEL_D,VEL_ACC,HEAD_MOT,HEAD_ACC,P_DOP
  ```
</details>

### Configuration
<details>
  <summary>Click to expand</summary>
  The config folder contains example configuration files. Please read the documentation of the
  individual parameters in the yaml file carefully. You have various options to trade-off accuracy and
  computational expense as well as to enable online calibration.

  #### V(I) 
  In case you want to run Visual(-Inertial) only, only `okvis2x_app_synchronous` and only the state estimator config is needed. Disabling the inertial part can be done by setting `imu_parameters.use` to `false` in the `okvis2-config`

  #### GNSS
  If you want to fuse global position measurements (e.g. from GNSS), you can do so by adding a `gps_parameters` block to the config file. It should contain the following parameters: 
  ```yaml
  gps_parameters:
    data_type: cartesian # cartesian or geodetic
    r_SA: [0.04943289490451834, 0.014787790366790175, 0.6079887122447304] # Antenna {A} offset in IMU {S} frame
    yaw_error_threshold: 0.1 # Threshold on yaw observability to freeze the optimization of the extrinsics (to enable the global alignment strategies mentioned in the paper)
    robust_gps_init: false # Use a robustified version of the Initialization (might be needed for low-grade GPS sensor measurements)
  ```
  If this block is specified in the config file, it will try to fuse GNSS measurements. Otherwise not.

  Note: The `DatasetReaderBase` class defines a variable `GNSS_LEAP_NANOSECONDS` which is used in the dataset readers to account for GNSS Leap seconds if needed (Offset of GNSS (usually atomic) clock to UTC). It is by default set to 0 but in case you need to consider them in your dataset, set it to the correct value (currently 18e+9).

  #### Submapping (Depth, Stereo Depth, Depth Fusion or LiDAR)
  You can enable the fusion of submap alignment factors by enabling `use_map_to_map_factors` or `use_map_to_live_factors` in the mapping config file.  

  When running OKVIS2-X with LiDAR and submap alignment, you also have to add a lidar block in the estimator config. This can look as follows:
  ```yaml
  # Please have a look examples in the HILTI and VBR config files.
  lidar:
    elevation_resolution_angle: 1.0 
    azimuth_resolution_angle:   0.18
    # Extrinsics IMU {S} <-> LiDAR {L}
    T_SL: [ 0.0, -1.0, 0., -0.0007,
          -1.0, 0.0, 0.0, -0.0086,
            0.0, 0.0, -1.0, 0.0550,
            0.0, 0.0, 0.0, 1.0]
  ```

  When running OKVIS2-X with depth network and submap alignment, please set corresponding parameters in `okvis2-config` file. 
  ```yaml
  # Please have a look examples in the EuRoC, HILTI, and VBR config files.
  cameras:
      - ...
          mapping: true,
          mapping_rectification: true,
        ...
  camera_parameters:
      ...
      deep_stereo_indices: [0,1]
      fov_scale: 1.0
  ```

  When running **FindAnything**, you have to ensure that there is a camera defined to have RGB data and a camera that has depth information. Please check our wiki for more information! 

  #### Pretrained models for depth networks
  We provide pretrained models, which predict both depth and its pixel-wise uncertainty, for the stereo network based on [Unimatch](https://github.com/autonomousvision/unimatch) and multi-view stereo network based on [SimpleRecon](https://github.com/nianticlabs/simplerecon). Please note that we provide two versions of the stereo network: `stereo-indoor-sigma.pt` and `stereo-mix-sigma.pt`. Please specify which model you would use in the top-level CMakeLists.txt. We used `stereo-indoor-sigma.pt` for the EuRoC and Hilti-Oxford dataset and `stereo-mix-sigma.pt` for the VBR dataset in the paper's evaluation.

  ```bash
    # Please set your desiered stereo depth model (DEPTH_FAST_FILE or DEPTH_ACCURACY_FILE) here.
    file(RENAME ${DEPTH_FAST_FILE} ${DEPTH_FILE})
  ```

  #### Pretrained models for open-vocabulary mapping
  We provide pretrained models for **FindAnything**. These are [eSAM](https://github.com/yformer/EfficientSAM) for mask proposals, and Mask-CLIP, based on the implementation from [f3rm](https://github.com/f3rm/f3rm). It is important to note that these can be exchanged for any model desired by the user as long as the format is the same as for the networks that we provide. 

  #### Online Calibration of Camera-IMU extrinsics
  You can enable online calibration of the Camera-IMU extrinsics by setting the respective parameters of the `online_calibration` parameters block in the `okvis-config`.
</details>

### Outputs and frames
<details>
  <summary>Click to expand</summary>
  **Fundamentally, the library is meant to be used in a way that you maintain an `okvis::Trajectory`
  object in your client code via two callbacks that you need to implement (and where it is your
  responsibility to ensure threadsafe access and modification):**

  1. Implement the callback that is called by the estimator whenever there are updates. The updates
    may involve many states, especially upon loop closure. Register it with
    `okvis::ViInterface::setOptimisedGraphCallback` and let it trigger the update to the
    `okvis::Trajectory` via `okvis::Trajectory::update`.
  2. If you would like to have the high-rate, most up-to-date states available via prediction with the
    newest IMU measurements, then also register an IMU callback (last) to
    `okvis::ViSensorBase::setImuCallback` and let it trigger the update to the
    `okvis::Trajectory` via `okvis::Trajectory::addImuMeasurement`.

  This mechanism allows availability of a consistent trajectory, `okvis::Trajectory`, on which you can
  query states for any time (at estimated, image timestamps, or otherwise).

  In terms of coordinate frames and notation,

  * `W` denotes the OKVIS World frame $`\underrightarrow{\mathcal{F}}_W`$ (z up)
  * `C_i` denotes the i-th camera frame $`\underrightarrow{\mathcal{F}}_{C_i}`$
  * `S` denotes the IMU sensor frame $`\underrightarrow{\mathcal{F}}_S`$
  * `B` denotes a (user-specified / robot) body frame $`\underrightarrow{\mathcal{F}}_B`$

  States contain the pose `T_WS` ($`\boldsymbol{T}_{WS}`$) as a position `r_WS`
  ($`_W\mathbf{r}_{S}`$) and quaternion `q_WS` ($`\mathbf{q}_{WS}`$), followed by the velocity
  in World frame `v_W` ($`_{W}\mathbf{v}`$) and gyro biases `b_g` ($`\mathbf{b}_g`$) as well as
  accelerometer biases `b_a` ($`\mathbf{b}_a`$).

  **See the example applications to understand how to use the estimator, sensor, callbacks, etc!**

  Furthermore, for every created submap a mesh file is exported (*.ply format; can be viewed e.g. in ParaView). The result directory can be specified via the command line when running the application.
</details>

### Building the project with ROS2 ###
<details>
  <summary>Click to expand</summary>
  Create a workspace e.g.

      mkdir -p ~/okvis_ws/src
      cd ~/okvis_ws/src

  Now you can clone the repo

      git clone --recurse-submodules git@github.com:ethz-mrl/OKVIS2-X.git

  Note: to successfully build and run the OKVIS2X nodes, you also need the following packages: `language_feature_node` , `language_feature_msgs` (all in our GitHub)

  You may want to run rosdep to make sure all dependencies are installed.

  Next, you can build:

      cd ~/okvis_ws
      colcon build

  Make sure to turn on the ROS2 specific build: it's on by default, but you may add
  `--cmake-args -DBUILD_ROS2=ON` to force it.

  For depth network, you need `--cmake-args -DUSE_NN=ON`, which is already turned on as default. If you use depth or lidar you can set `--cmake-args -DUSE_NN=OFF`.

  If you don't need color fusion, you can set `--cmake-args -DUSE_COLIDMAP=OFF`, which is `ON` as default. For example,

      # With depth sensor with color fusion
      colcon build --cmake-args -DUSE_NN=OFF -DUSE_COLIDMAP=ON
      
      # With depth network without color fusion
      colcon build --cmake-args -DUSE_NN=ON -DUSE_COLIDMAP=OFF

      # With networks (eSAM and CLIP) and color fusion, a requirement for FindAnything
      colcon build --cmake-args -DUSE_NN=ON -DUSE_COLIDMAP=ON

      # With depth sensor or lidar without color fusion
      colcon build --cmake-args -DUSE_NN=OFF -DUSE_COLIDMAP=OFF

  Currently, the lidar nodes only support `-DUSE_COLIDMAP=OFF` (colour fusion is not implemented).

  #### Running ROS2 nodes ####

  In this repository, the *pure OKVIS* nodes can be run by running the OKVIS2-X nodes in V(I)-only mode by setting parameters in the config file (i.e. **parameters.output.enable_submapping = false** or additionally **parameters.imu.use=false** for Vision Only).

  We kept the realsense publisher which only publishes realsense sensor streams as ros2 topics:

      ros2 launch okvis okvis_node_realsense_publisher.launch.xml config_filename:=<config.yaml>

  Note that you may use `okvis_node_realsense_publisher` to record a
  bag, minimally with

      ros2 bag record /okvis/imu0 /okvis/cam0/image_raw /okvis/cam1/image_raw

  #### OKVIS2-X ROS2 nodes ####

  There are two type of nodes: subscriber nodes and realsense nodes.

  For the subscriber nodes, there are 5 different modes: `lidar`, `depth_image` (ros2 topic providing depth images), `stereo_network` (Stereo Depth Prediction), `depth_fusion` (MVS Depth Fusion), `vision_language` (FindAnything)

  ```bash
  ros2 launch okvis okvis2x_node_subscriber.launch.xml config_filename:=[config] se_config_filename:=[se2_config] \ 
      [lidar/depth_image/stereo_network/depth_fusion/vision_language]:=true
  ```

  For the realsense nodes, there are 4 different modes: `depth_image` (ros2 topic providing depth images), `stereo_network` (Stereo Depth Prediction), `depth_fusion` (MVS Depth Fusion), `vision_language` (FindAnything)
  ```bash
  ros2 launch okvis okvis2x_node_realsense.launch.xml config_filename:=[config] se_config_filename:=[se2_config] \ 
      [lidar/depth_image/stereo_network/depth_fusion/vision_language]:=true
  ```

  For the VI-only mode (OKVIS2-X without submapping), please set `enable_submapping=false` in `[config]` and launch with `depth_image:=true`. This will fall back to OKVIS2.
  ```bash
  ros2 launch okvis okvis2x_node_subscriber.launch.xml config_filename:=[config] se_config_filename:=[se2_config] \
      depth_image:=true
  ```

  **Notes:**

  * **IMPORTANT**: never forget to set `<set_env name="OMP_NUM_THREADS" value="2"/>`  (in the launch file ideally), otherwise depth integration might be horribly slow in SE2
  * There are additional options to configure saving of meshes at the end or the final trajectory (`save_submap_meshes`, `csv_path`,...). This can be triggered by calling the following ROS2 Service:
  ```bash
  ros2 service call /okvis/shutdown std_srvs/srv/SetBool
  ```
  * To run the **FindAnything** nodes you need additionally the `language_feature_node` and `language_feature_msgs` (for text queries) in your workspace: https://github.com/ethz-mrl/language_feature_node and https://github.com/ethz-mrl/language_feature_msgs.
</details>


### HEALTH WARNING: calibration ###
<details>
  <summary>Click to expand</summary>
  If you would like to run the software/library on your own hardware setup, be aware that good results
  (or results at all) may only be obtained with appropriate calibration of the

  * camera intrinsics,
  * camera extrinsics (poses relative to the IMU),
  * knowledge about the IMU noise parameters,
  * and ACCURATE TIME SYNCHRONISATION OF ALL SENSORS.

  To perform a calibration yourself, we recommend using [kalibr](https://github.com/ethz-asl/kalibr).
</details>


### Contribution guidelines ###

* Contact one of the main authors (simon.boche@tum.de, jaehyung.jung@tum.de or sebastian.barbas@tum.de) to request access to the github repository.

* Programming guidelines: please follow
  https://github.com/ethz-asl/programming_guidelines/wiki/Cpp-Coding-Style-Guidelines.

* Writing tests: please write unit tests (gtest).

* Code review: please create a pull request for all changes proposed. The pull request will be
  reviewed by an admin before merging.

### Support ###

The developpers will be happy to assist you or to consider bug reports / feature requests. Please use the Issues section of the repository. In case of bug reports, please consider also providing config files or dataset details that allow us to reproduce the issue.
