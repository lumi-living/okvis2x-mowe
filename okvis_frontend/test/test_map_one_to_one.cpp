/**
 * @file test_map_one_to_one.cpp
 * @brief T-0129: one keypoint per landmark (keepBestKeypointPerLandmark), ratio test.
 */
#include <gtest/gtest.h>

#include <limits>

#include <okvis/MapMatchOneToOne.hpp>

using okvis::LandmarkId;

TEST(MapOneToOne, KeepsLowestDistancePerLandmark) {
  // keypoints 0,2,3 all chose landmark 7 (neighbouring XFeat keypoints), 1 chose 9, 4 none
  std::vector<LandmarkId> ids = {LandmarkId(7), LandmarkId(9), LandmarkId(7), LandmarkId(7),
                                 LandmarkId()};
  const std::vector<double> dist = {0.20, 0.10, 0.05, 0.12, 0.25};
  EXPECT_EQ(okvis::keepBestKeypointPerLandmark(ids, dist), 2u);
  EXPECT_FALSE(ids[0].isInitialised());
  EXPECT_EQ(ids[1].value(), 9u);
  EXPECT_EQ(ids[2].value(), 7u);
  EXPECT_FALSE(ids[3].isInitialised());
  EXPECT_FALSE(ids[4].isInitialised());
}

TEST(MapOneToOne, DistinctLandmarksUntouchedAndTieKeepsFirst) {
  std::vector<LandmarkId> ids = {LandmarkId(1), LandmarkId(2), LandmarkId(2)};
  EXPECT_EQ(okvis::keepBestKeypointPerLandmark(ids, {0.1, 0.2, 0.2}), 1u);
  EXPECT_EQ(ids[0].value(), 1u);
  EXPECT_EQ(ids[1].value(), 2u);
  EXPECT_FALSE(ids[2].isInitialised());
}

TEST(MapOneToOne, RatioTestDropsAmbiguousOnly) {
  const double inf = std::numeric_limits<double>::max();
  std::vector<LandmarkId> ids = {LandmarkId(1), LandmarkId(2), LandmarkId(3), LandmarkId()};
  // k0: clear (0.05 vs 0.20, L2 ratio 0.5); k1: ambiguous (0.10 vs 0.11); k2: no rival
  const std::vector<double> d = {0.05, 0.10, 0.10, 0.25}, d2 = {0.20, 0.11, inf, 0.01};
  std::vector<LandmarkId> off = ids;
  EXPECT_EQ(okvis::rejectAmbiguousMatches(off, d, d2, 1.0), 0u);  // disabled
  EXPECT_EQ(okvis::rejectAmbiguousMatches(ids, d, d2, 0.9), 1u);
  EXPECT_EQ(ids[0].value(), 1u);
  EXPECT_FALSE(ids[1].isInitialised());
  EXPECT_EQ(ids[2].value(), 3u);
}
