# okvis_apps — colcon anchor for the OKVIS2-X dataset apps (mow-e T-0113).
#
# The executables in src/ are built by the top-level okvis CMakeLists (BUILD_APPS)
# and cannot be built standalone: okvis::MultisensorProcessing's exported link
# interface names SRL::Supereight2, a superbuild-internal target that is not
# exported. This package depends on okvis (so the apps get built and installed
# first) and installs the Python helper scripts next to them.
cmake_minimum_required(VERSION 3.16)
project(okvis_apps NONE)
install(PROGRAMS scripts/bag_creator.py DESTINATION lib/${PROJECT_NAME})
