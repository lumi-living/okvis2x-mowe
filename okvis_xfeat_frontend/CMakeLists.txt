# okvis_xfeat_frontend — XFeat-on-TensorRT feature frontend bridge.
#
# Bridges mowe_camera_core (capture) → TensorRT (XFeat) → okvis::xfeat features.
# Buildable two ways:
#   * as an OKVIS submodule: add_subdirectory() from the top CMakeLists behind
#     option(USE_MOWE_XFEAT) — Eigen3::Eigen already exists there.
#   * standalone: cmake -S okvis_xfeat_frontend -B build  (finds Eigen3 itself).
#
# mowe_camera_core is found via its vanilla CMake package config (ament-free).
cmake_minimum_required(VERSION 3.16)
project(okvis_xfeat_frontend CXX)

option(USE_TENSORRT "Build with the real TensorRT/CUDA inference path" OFF)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# --- dependencies ---
find_package(mowe_camera_core CONFIG REQUIRED)

add_library(okvis_xfeat_frontend
  src/XFeatFrontend.cpp
  src/TensorRTEngine.cpp
  src/LighterGlueMatcher.cpp)
add_library(okvis::XFeatFrontend ALIAS okvis_xfeat_frontend)

target_include_directories(okvis_xfeat_frontend
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(okvis_xfeat_frontend
  PUBLIC
    mowe_camera::core)
target_compile_features(okvis_xfeat_frontend PUBLIC cxx_std_17)

if(USE_TENSORRT)
  enable_language(CUDA)
  # Jetson Orin Nano (Ampere) is compute capability 8.7.
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 87)
  endif()
  find_package(CUDAToolkit REQUIRED)
  find_library(NVINFER_LIB nvinfer REQUIRED)

  # The CUDA kernel lives in its own CUDA-C++17 target, deliberately NOT linking
  # mowe_camera::core. The core exports C++20 (std::span) publicly; if the .cu
  # were part of the main library that requirement propagates to CUDA and forces
  # a CUDA20 dialect nvcc/CMake can't map with extensions on. The kernel needs no
  # C++20, so isolating it keeps it at CUDA C++17 and sidesteps the clash.
  add_library(okvis_xfeat_cuda STATIC src/preprocess.cu)
  set_target_properties(okvis_xfeat_cuda PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON)
  target_include_directories(okvis_xfeat_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(okvis_xfeat_cuda PRIVATE CUDA::cudart)

  target_link_libraries(okvis_xfeat_frontend
    PRIVATE okvis_xfeat_cuda CUDA::cudart ${NVINFER_LIB})
  target_compile_definitions(okvis_xfeat_frontend PRIVATE OKVIS_XFEAT_USE_TENSORRT=1)
  message(STATUS "okvis_xfeat_frontend: TensorRT inference ENABLED (sm_${CMAKE_CUDA_ARCHITECTURES})")
else()
  message(STATUS "okvis_xfeat_frontend: stub mode (no TensorRT) — features empty")
endif()

# When built inside the OKVIS superbuild, okvis_frontend links this library
# PRIVATE — but okvis_frontend is STATIC, so the dependency lands in its
# installed interface as $<LINK_ONLY:...> and these targets must join the
# okvisTargets export set (okvisConfig.cmake.in then find_dependency()s
# mowe_camera_core). Standalone builds skip this (no okvisTargets export).
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  include(GNUInstallDirs)
  set(_xfeat_export_targets okvis_xfeat_frontend)
  if(TARGET okvis_xfeat_cuda)
    list(APPEND _xfeat_export_targets okvis_xfeat_cuda)
  endif()
  install(TARGETS ${_xfeat_export_targets} EXPORT okvisTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
  install(DIRECTORY include/okvis DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp")
endif()

# Skeleton end-to-end demo (camera → frontend). No ROS.
add_executable(xfeat_frontend_demo src/xfeat_frontend_demo.cpp)
target_link_libraries(xfeat_frontend_demo PRIVATE okvis_xfeat_frontend)

# Optional keypoint visualization (draws kpts, saves PNG / imshow). OpenCV ships
# with JetPack; if missing the demo still builds (prints counts only).
find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs highgui)
if(OpenCV_FOUND)
  target_include_directories(xfeat_frontend_demo PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(xfeat_frontend_demo PRIVATE ${OpenCV_LIBS})
  target_compile_definitions(xfeat_frontend_demo PRIVATE MOWE_XFEAT_DEMO_OPENCV=1)
  message(STATUS "okvis_xfeat_frontend: demo viz ENABLED (OpenCV ${OpenCV_VERSION})")
else()
  message(STATUS "okvis_xfeat_frontend: demo viz disabled (OpenCV not found)")
endif()
