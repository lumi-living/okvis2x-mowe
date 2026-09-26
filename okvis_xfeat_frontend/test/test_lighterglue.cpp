// CPU-only gtest for the LighterGlue staging/decoding helpers (T-0114):
// top-K selection, validity-mask construction from scores, index-pair
// decoding on synthetic engine outputs. Runs on host, qemu and device.
#include <gtest/gtest.h>

#include <vector>

#include "okvis/xfeat/LighterGlueUtil.hpp"

using namespace okvis::xfeat;

namespace {
struct Set {
  std::vector<float> kp, sc;
};
// n keypoints along a diagonal with scores descending in index order.
Set make(std::size_t n, float top = 0.9f) {
  Set s;
  for (std::size_t i = 0; i < n; ++i) {
    s.kp.push_back(10.f * float(i));
    s.kp.push_back(5.f * float(i));
    s.sc.push_back(top - 0.001f * float(i));
  }
  return s;
}
}  // namespace

TEST(LighterGlueUtil, TopKPicksHighestScoresDescending) {
  const float sc[] = {0.1f, 0.9f, 0.5f, 0.7f, 0.3f};
  const auto o = top_k_by_score(sc, 5, 3);
  ASSERT_EQ(o.size(), 3u);
  EXPECT_EQ(o[0], 1u);
  EXPECT_EQ(o[1], 3u);
  EXPECT_EQ(o[2], 2u);
  const auto all = top_k_by_score(sc, 5, 8);  // n < k: all rows, still sorted
  ASSERT_EQ(all.size(), 5u);
  EXPECT_EQ(all[0], 1u);
  EXPECT_EQ(all[4], 0u);
}

TEST(LighterGlueUtil, StageMasksUnusedSlotsAndNormalises) {
  const Set s = make(3);
  StagedSet st;
  stage_set(s.kp.data(), s.sc.data(), 3, 640, 384, 8, st);
  EXPECT_EQ(st.used, 3u);
  EXPECT_EQ(st.capacity(), 8u);
  EXPECT_EQ(st.masked_slots(), 5u);  // exactly the padded tail is masked
  for (std::size_t i = 3; i < 8; ++i) {
    EXPECT_EQ(st.scores[i], kLighterGluePadScore);
    EXPECT_EQ(st.kpts_norm[i * 2], 0.f);
  }
  // kornia normalize_keypoints: (xy - size/2) / (max/2); px (0,0) -> (-1, -0.6)
  EXPECT_FLOAT_EQ(st.kpts_norm[0], -1.f);
  EXPECT_FLOAT_EQ(st.kpts_norm[1], -0.6f);
  EXPECT_FLOAT_EQ(st.scores[0], 0.9f);
}

TEST(LighterGlueUtil, StageTruncatesToTopKAndMasksNothing) {
  Set s = make(10);
  s.sc[7] = 1.0f;  // a late strong row must make the cut
  StagedSet st;
  stage_set(s.kp.data(), s.sc.data(), 10, 640, 384, 4, st);
  EXPECT_EQ(st.used, 4u);
  EXPECT_EQ(st.masked_slots(), 0u);
  EXPECT_EQ(st.src_index[0], 7u);
  EXPECT_EQ(st.src_index[1], 0u);
  EXPECT_FLOAT_EQ(st.kpts_norm[0], (70.f - 320.f) / 320.f);  // row 7 in slot 0
}

TEST(LighterGlueUtil, CallerRowsWithNonPositiveScoreAreMasked) {
  Set s = make(3);
  s.sc[1] = 0.f;  // engine treats score <= 0 as padding (export.py: valid = score > 0)
  StagedSet st;
  stage_set(s.kp.data(), s.sc.data(), 3, 640, 384, 3, st);
  EXPECT_EQ(st.masked_slots(), 1u);
  const std::int32_t m[] = {0, 1, 2};
  const float ms[] = {0.9f, 0.9f, 0.9f};
  const auto d = decode_matches(m, ms, st, st, 0.1f);
  EXPECT_EQ(d.size(), 2u);  // the masked row can neither match nor be matched
}

TEST(LighterGlueUtil, DecodeMapsSlotsBackToCallerIndices) {
  // A has 4 rows, B has 3; both truncated/reordered by score into K=4 slots.
  Set a = make(4), b = make(3);
  a.sc[3] = 1.0f;  // A order: 3,0,1,2
  b.sc[2] = 1.0f;  // B order: 2,0,1
  StagedSet sa, sb;
  stage_set(a.kp.data(), a.sc.data(), 4, 640, 384, 4, sa);
  stage_set(b.kp.data(), b.sc.data(), 3, 640, 384, 4, sb);
  // Engine: slot0->slot1, slot1->none, slot2->slot0, slot3->slot3 (padded B!)
  const std::int32_t m[] = {1, -1, 0, 3};
  const float ms[] = {0.8f, 0.0f, 0.05f, 0.9f};
  const auto d = decode_matches(m, ms, sa, sb, 0.1f);
  ASSERT_EQ(d.size(), 1u);  // slot2 below min_score, slot3 targets padding
  EXPECT_EQ(d.indices[0].first, 3u);   // A slot0 = caller row 3
  EXPECT_EQ(d.indices[0].second, 0u);  // B slot1 = caller row 0
  EXPECT_FLOAT_EQ(d.scores[0], 0.8f);
  const auto loose = decode_matches(m, ms, sa, sb, 0.0f);
  EXPECT_EQ(loose.size(), 2u);
  EXPECT_EQ(loose.indices[1].first, 1u);   // A slot2 = caller row 1
  EXPECT_EQ(loose.indices[1].second, 2u);  // B slot0 = caller row 2
}

TEST(LighterGlueUtil, DecodeRejectsOutOfRangeTargets) {
  const Set a = make(2), b = make(2);
  StagedSet sa, sb;
  stage_set(a.kp.data(), a.sc.data(), 2, 640, 384, 4, sa);
  stage_set(b.kp.data(), b.sc.data(), 2, 640, 384, 4, sb);
  const std::int32_t m[] = {7, -2, 0, 1};  // garbage, negative, padded A slots
  const float ms[] = {1.f, 1.f, 1.f, 1.f};
  EXPECT_EQ(decode_matches(m, ms, sa, sb, 0.f).size(), 0u);
}
