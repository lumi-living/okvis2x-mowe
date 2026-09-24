# mow-e (okvis2x-mowe) initial-cache file.
#
# Preloads the build options mow-e uses WITHOUT changing the upstream OKVIS2-X
# option() defaults in the top-level CMakeLists.txt, so rebasing onto a newer
# OKVIS2-X never conflicts on them. Use it as
#
#   cmake -C cmake/mowe-defaults.cmake -S . -B build [-DUSE_MOWE_XFEAT=ON ...]
#   colcon build --cmake-args -C "$PWD/src/okvis2x-mowe/cmake/mowe-defaults.cmake" ...
#
# Anything passed with -D on the same command line still wins (it is applied
# after this file). See README_mowe_bringup.md for the full flag table.

# No LibTorch: the depth / MVS / FindAnything networks are not used on mow-e,
# and USE_NN=ON also downloads models from cvg.cit.tum.de at configure time.
set(USE_NN OFF CACHE BOOL "Use keypoint classification as part of okvis, requires torch")
set(USE_GPU OFF CACHE BOOL "Use keypoint classification with GPU inference, requires torch with Cuda")
# No colour / object-ID (FindAnything) mapping.
set(USE_COLIDMAP OFF CACHE BOOL "Whether to build okvis supportin colour and object's id")
# No Intel RealSense on the robot (OV9281 stereo + SCH16T IMU).
set(HAVE_LIBREALSENSE OFF CACHE BOOL "Use realsense as part of okvis")
# Keep tests / supereight2 extras out of the robot build.
set(BUILD_TESTS OFF CACHE BOOL "Builds all gtests")
set(SE_TEST OFF CACHE BOOL "Compile the supereight tests")
set(SE_APP OFF CACHE BOOL "Compile the supereight application")
# The XFeat/LighterGlue frontend stays opt-in (-DUSE_MOWE_XFEAT=ON), because it
# needs mowe_camera_core and, for real inference, CUDA + TensorRT.
set(USE_MOWE_XFEAT OFF CACHE BOOL "Build the Mow-e XFeat feature frontend bridge")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build.")
endif()
