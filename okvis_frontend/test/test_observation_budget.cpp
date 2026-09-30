// T-0140: per-image observation budget (ObservationBudget.hpp).
#include <algorithm>
#include <random>
#include <gtest/gtest.h>
#include <okvis/ObservationBudget.hpp>

using okvis::BudgetCandidate;
using okvis::selectObservationSurplus;

namespace {
// A synthetic window image: 900 observations, clustered in the left third, mixed quality.
std::vector<BudgetCandidate> window(unsigned seed = 1) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::vector<BudgetCandidate> c;
  for (size_t k = 0; k < 900; ++k) {
    const double x = k < 600 ? u(rng) * 170.0 : 170.0 + u(rng) * 342.0;
    c.push_back({k, x, u(rng) * 512.0, std::floor(u(rng) * 4.0) / 4.0,
                 size_t(2 + rng() % 6), 1000 + k});
  }
  return c;
}
}  // namespace

TEST(ObservationBudget, OffOrUnderBudgetKeepsAll) {
  EXPECT_TRUE(selectObservationSurplus(window(), 512, 512, 0).empty());
  EXPECT_TRUE(selectObservationSurplus(window(), 512, 512, 900).empty());
}

TEST(ObservationBudget, KeepsAtMostNAndSpreadsOverTheImage) {
  const auto c = window();
  for (size_t n : {400u, 300u, 200u, 150u}) {
    const auto drop = selectObservationSurplus(c, 512, 512, n);
    ASSERT_EQ(c.size() - drop.size(), n);
    // the dense left third holds 2/3 of the candidates but gets roughly its area share
    size_t keptLeft = 0;
    for (const auto& x : c)
      if (!std::binary_search(drop.begin(), drop.end(), x.keypoint) && x.u < 170.0) ++keptLeft;
    EXPECT_LT(keptLeft, n / 2) << n;
  }
}

TEST(ObservationBudget, BestQualityPerCellWins) {
  // one cell, three candidates: quality, then observation count, then landmark id
  std::vector<BudgetCandidate> c = {{0, 10, 10, 0.5, 3, 7}, {1, 11, 10, 0.9, 2, 9},
                                    {2, 12, 10, 0.5, 3, 5}, {3, 13, 10, 0.5, 4, 8}};
  EXPECT_EQ(selectObservationSurplus(c, 512, 512, 3), (std::vector<size_t>{0}));
  EXPECT_EQ(selectObservationSurplus(c, 512, 512, 1), (std::vector<size_t>{0, 2, 3}));
}

TEST(ObservationBudget, DeterministicUnderInputOrder) {
  auto c = window(3);
  const auto ref = selectObservationSurplus(c, 512, 512, 200);
  std::mt19937 rng(42);
  for (int i = 0; i < 5; ++i) {
    std::shuffle(c.begin(), c.end(), rng);
    EXPECT_EQ(selectObservationSurplus(c, 512, 512, 200), ref);
  }
}
