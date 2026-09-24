cmake_minimum_required(VERSION 3.8...3.16)

# Copy file SOURCE to DESTINATION if DESTINATION exists and show an error
# message otherwise.
function(copy_if_exists SOURCE DESTINATION)
  if(EXISTS "${DESTINATION}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${SOURCE}" "${DESTINATION}")
  else()
    message(FATAL_ERROR "File ${DESTINATION} doesn't exist. Did you forget to update the git submodules? Run:\ngit submodule update --init --recursive\n")
  endif()
endfunction()

if(NOT USE_SYSTEM_BRISK)
  # mow-e (okvis2x-mowe): BRISK chooses NEON vs. -mssse3 from
  # CMAKE_HOST_SYSTEM_PROCESSOR, i.e. the *build* machine. When cross-compiling
  # for the Jetson (aarch64) on an x86_64 host it therefore picks -mssse3 and
  # the build fails. patches/brisk/CMakeLists.txt is the pinned brisk
  # CMakeLists.txt (1ef8b42) with that one test switched to
  # CMAKE_SYSTEM_PROCESSOR (the target); see patches/brisk/cross-compile.patch.
  # It is copied over the submodule's file at configure time, exactly like the
  # DBoW2 and OpenGV patches below, so it survives `git submodule update`.
  # The SHA-256 guard refuses to overwrite a brisk CMakeLists.txt that is
  # neither the pinned upstream file nor the already-patched one (i.e. the
  # submodule was bumped and the patch copy must be regenerated).
  set(_mowe_brisk_cml "${CMAKE_CURRENT_SOURCE_DIR}/brisk/CMakeLists.txt")
  set(_mowe_brisk_upstream_sha256
      "c3836e8e4a7f1dd08caa7b967870fa46a1940e1d0c7d4590b5dd7f7d0c6d19de")
  set(_mowe_brisk_patched_sha256
      "87b124208137424a705bfd6eb772f52c320b484a3bbec7889e6809f949b4364d")
  if(EXISTS "${_mowe_brisk_cml}")
    file(SHA256 "${_mowe_brisk_cml}" _mowe_brisk_sha256)
    if(_mowe_brisk_sha256 STREQUAL _mowe_brisk_upstream_sha256)
      copy_if_exists(${CMAKE_CURRENT_SOURCE_DIR}/patches/brisk/CMakeLists.txt ${_mowe_brisk_cml})
      message(STATUS "mow-e: applied BRISK cross-compile patch (CMAKE_SYSTEM_PROCESSOR)")
    elseif(NOT _mowe_brisk_sha256 STREQUAL _mowe_brisk_patched_sha256)
      message(FATAL_ERROR "external/brisk/CMakeLists.txt is neither the pinned "
        "upstream file (brisk 1ef8b42) nor the mow-e patched copy. The brisk "
        "submodule was probably moved: regenerate external/patches/brisk/ "
        "(apply cross-compile.patch to the new file) and update the hashes here.")
    endif()
  else()
    # Lets copy_if_exists print the usual "did you init the submodules" error.
    copy_if_exists(${CMAKE_CURRENT_SOURCE_DIR}/patches/brisk/CMakeLists.txt ${_mowe_brisk_cml})
  endif()
  option(BRISK_BUILD_DEMO OFF)
  add_subdirectory(brisk)
endif()

if(NOT USE_SYSTEM_DBOW2)
  # Use a modified CMakeLists.txt for DBoW2 to allow building as a static
  # library and to fix issues when installing it along with okvis.
  copy_if_exists(${CMAKE_CURRENT_SOURCE_DIR}/patches/DBoW2/CMakeLists.txt ${CMAKE_CURRENT_SOURCE_DIR}/DBoW2/CMakeLists.txt)
  option(BUILD_Demo OFF)
  add_subdirectory(DBoW2)
  set(DBoW2_LIBRARIES DBoW2 PARENT_SCOPE)
endif()

if(NOT USE_SYSTEM_CERES)
  if(OPENMP_FOUND)
    set(CERES_THREADING_MODEL OPENMP)
  else()
    set(CERES_THREADING_MODEL CXX_THREADS)
  endif()
  set(BUILD_TESTING OFF CACHE BOOL "Build tests" FORCE)
  set(BUILD_EXAMPLES OFF CACHE BOOL "Build examples" FORCE)
  set(BUILD_BENCHMARKS OFF CACHE BOOL "Build benchmarks" FORCE)
  add_subdirectory(ceres-solver)
endif()

# Use a modified CMakeLists.txt for OpenGV to build with the same flags as
# okvis2.
copy_if_exists(${CMAKE_CURRENT_SOURCE_DIR}/patches/opengv/CMakeLists.txt ${CMAKE_CURRENT_SOURCE_DIR}/opengv/CMakeLists.txt)
add_subdirectory(opengv)
set(OPENGV_LIBRARIES opengv PARENT_SCOPE)

if(BUILD_TESTS)
  option(INSTALL_GTEST OFF)
  add_subdirectory(googletest)
endif()
