/**
 * @file ObservationBudget.hpp
 * @brief T-0140: per-image observation budget of the newest frame.
 *
 * 62 % of a realtime solve on the Orin Nano is evaluating ~9 100 reprojection residuals
 * (T-0128). Capping the observations each image contributes shrinks every iteration; the
 * surplus observations are removed (not set constant: a constant landmark still costs
 * its residual and pose Jacobian every iteration, research memo 2026-09-30 §3).
 * Selection is deterministic so blocking replay stays bit-identical (T-0131).
 */
#ifndef INCLUDE_OKVIS_OBSERVATIONBUDGET_HPP_
#define INCLUDE_OKVIS_OBSERVATIONBUDGET_HPP_

#include <algorithm>
#include <cstdint>
#include <vector>

namespace okvis {

/// \brief One observation of the newest frame in one image.
struct BudgetCandidate {
  size_t keypoint;     ///< keypoint index in the image
  double u, v;         ///< keypoint position [px]
  double quality;      ///< landmark quality (higher = better)
  size_t observations; ///< landmark observation count
  uint64_t landmarkId; ///< tiebreak
};

/// \brief Pick at most maxObs candidates spread over a cells x cells grid: rank per cell by
///        (quality desc, observations desc, landmarkId asc), then take rank 0 of every cell,
///        rank 1 of every cell, ...; a rank that does not fit whole goes by the same order.
///        Candidates outside the image are clamped into the border cells.
/// \return Keypoint indices of the candidates to DROP (surplus), sorted ascending.
inline std::vector<size_t> selectObservationSurplus(std::vector<BudgetCandidate> c,
                                                    double width, double height,
                                                    size_t maxObs, int cells = 8) {
  std::vector<size_t> drop;
  if (maxObs == 0 || c.size() <= maxObs) return drop;
  auto better = [](const BudgetCandidate& a, const BudgetCandidate& b) {
    if (a.quality != b.quality) return a.quality > b.quality;
    if (a.observations != b.observations) return a.observations > b.observations;
    return a.landmarkId < b.landmarkId;
  };
  auto cellOf = [&](const BudgetCandidate& x) {
    const int cx = std::clamp(int(x.u / width * cells), 0, cells - 1);
    const int cy = std::clamp(int(x.v / height * cells), 0, cells - 1);
    return cy * cells + cx;
  };
  std::vector<std::vector<BudgetCandidate>> grid(size_t(cells * cells));
  for (const auto& x : c) grid[size_t(cellOf(x))].push_back(x);
  size_t maxRank = 0;
  for (auto& g : grid) {
    std::sort(g.begin(), g.end(), better);
    maxRank = std::max(maxRank, g.size());
  }
  size_t kept = 0;
  for (size_t r = 0; r < maxRank; ++r) {
    std::vector<BudgetCandidate> tier;
    for (const auto& g : grid) if (r < g.size()) tier.push_back(g[r]);
    std::sort(tier.begin(), tier.end(), better);
    for (const auto& x : tier) {
      if (kept < maxObs) ++kept;
      else drop.push_back(x.keypoint);
    }
  }
  std::sort(drop.begin(), drop.end());
  return drop;
}

}  // namespace okvis

#endif  // INCLUDE_OKVIS_OBSERVATIONBUDGET_HPP_
