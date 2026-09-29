/**
 * mow-e (T-0131, mowe-nav-kb 08): blocking replay must be bit-deterministic.
 *
 * ViGraph builds its ceres::Problem with enable_fast_removal. Ceres 2.2.0's
 * ProblemImpl::RemoveParameterBlock then removed the block's residuals in
 * unordered_set<ResidualBlock*> order (= heap addresses), and every removal swaps
 * the last residual into the hole, so the program's residual order -- and the
 * floating-point summation order of every later solve -- depended on where malloc
 * had put things. Two replays of one binary diverged by ~1e-4 m (TUM-VI room1).
 * external/CMakeLists.txt patches the loop to remove in program-index order,
 * highest first, which is exactly what the slow (non-fast-removal) path does.
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */
#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include <ceres/ceres.h>

namespace {

/// r = x - tag; the tag identifies the residual in the program afterwards.
struct Tagged : public ::ceres::SizedCostFunction<1, 1> {
  explicit Tagged(int tag) : tag(tag) {}
  bool Evaluate(double const* const* x, double* r, double** J) const override {
    r[0] = x[0][0] - double(tag);
    if (J && J[0]) J[0][0] = 1.0;
    return true;
  }
  int tag;
};

/// Residual order of the program after removing a parameter block that still has
/// residuals whose program positions were shuffled by earlier removals. `pad`
/// scatters the heap between insertions so residual-block addresses differ per call.
std::vector<int> programOrderAfterRemoval(bool fastRemoval, size_t pad) {
  constexpr int N = 64;
  std::vector<std::unique_ptr<Tagged>> costs;
  std::vector<std::unique_ptr<char[]>> scatter;
  ::ceres::Problem::Options options;
  options.enable_fast_removal = fastRemoval;
  options.cost_function_ownership = ::ceres::DO_NOT_TAKE_OWNERSHIP;
  ::ceres::Problem problem(options);
  double x = 0.0;
  std::vector<double> y(N, 0.0);
  std::vector<::ceres::ResidualBlockId> onY;
  auto add = [&](int tag, double* block) {
    costs.emplace_back(new Tagged(tag));
    if (pad) scatter.emplace_back(new char[pad * size_t(1 + (tag * 7919) % 13)]);
    return problem.AddResidualBlock(costs.back().get(), nullptr, block);
  };
  for (int i = 0; i < N; ++i) {
    add(i, &x);
    onY.push_back(add(1000 + i, &y[i]));
  }
  for (int i = 1; i < N; i += 3) problem.RemoveResidualBlock(onY[i]);  // swap-with-last shuffles x's residuals
  for (int i = N; i < N + 16; ++i) add(i, &x);
  problem.RemoveParameterBlock(&x);

  std::vector<::ceres::ResidualBlockId> blocks;
  problem.GetResidualBlocks(&blocks);
  std::vector<int> order;
  for (auto id : blocks) {
    order.push_back(static_cast<const Tagged*>(problem.GetCostFunctionForResidualBlock(id))->tag);
  }
  return order;
}

}  // namespace

TEST(okvisTestSuite, CeresFastRemovalIsAddressIndependent) {
  const std::vector<int> reference = programOrderAfterRemoval(false, 0);  // slow path: program order
  ASSERT_FALSE(reference.empty());
  for (size_t pad : {size_t(0), size_t(24), size_t(200), size_t(4096)}) {
    EXPECT_EQ(programOrderAfterRemoval(true, pad), reference) << "heap pad " << pad;
  }
}
