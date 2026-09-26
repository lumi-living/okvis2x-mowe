# okvis_apps

Dataset / bench applications for OKVIS2-X (`okvis_app_synchronous`, `okvis2x_app_synchronous`, `dbow2_test`, RealSense apps).

The binaries are compiled by the **top-level `okvis` superbuild** (`-DBUILD_APPS=ON`, the default in `tools/cross/colcon*.meta`). This directory is also a colcon package (`package.xml`, plain CMake) that depends on `okvis`, so `colcon build --packages-up-to okvis_apps` builds and installs the apps; on its own it only installs `scripts/bag_creator.py`. A standalone build against `find_package(okvis)` is not possible today because the exported `okvis::MultisensorProcessing` link interface references the non-exported `SRL::Supereight2` target (mow-e T-0113).

`okvis_app_synchronous <config.yaml> <dataset>/mav0/ [save-dir]` writes the causal / final trajectories and, since T-0113, `frontend_stats.json` (keypoints per frame, stereo and map matches, front-end ms mean/p99, `engine_loaded`, loop closures) to the save dir.
