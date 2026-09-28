/**
 * @file KeypointGrid.hpp
 * @brief Coarse image grid over keypoints for the 3D map matching (T-0128).
 *
 * `Frontend::matchToMapByThread` gates every (landmark, keypoint) pair on the
 * image distance between the landmark's predicted projection (IMU-propagated
 * pose) and the keypoint before it scores descriptors. Scanning all keypoints
 * per landmark is O(landmarks x keypoints); bucketing the keypoints in cells of
 * the gate radius makes each landmark visit only the 2x2..3x3 cells its gate
 * disk overlaps. The visited set is a superset of the keypoints inside the
 * gate, so with the caller's exact distance test kept the matches are identical
 * to the full scan.
 */
#ifndef INCLUDE_OKVIS_KEYPOINTGRID_HPP_
#define INCLUDE_OKVIS_KEYPOINTGRID_HPP_

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <vector>

namespace okvis {

class KeypointGrid {
public:
  /// \brief Bucket keypoints.col(k) for k in [begin, end) with use[k] into
  /// cells of
  ///        cellSize [px] (use the gate radius). Keypoints outside
  ///        [0,width)x[0,height) land in the border cells.
  KeypointGrid(const Eigen::Matrix2Xd &keypoints, size_t begin, size_t end,
               const std::vector<bool> &use, double width, double height,
               double cellSize)
      : cell_(std::max(cellSize, 1.0)),
        nu_(std::max(1, int(std::ceil(width / cell_)))),
        nv_(std::max(1, int(std::ceil(height / cell_)))),
        start_(size_t(nu_ * nv_) + 1, 0) {
    std::vector<int> cellOf(end > begin ? end - begin : 0, -1);
    for (size_t k = begin; k < end; ++k) {
      if (!use[k])
        continue;
      const int c = cellIndex(keypoints(0, k), keypoints(1, k));
      cellOf[k - begin] = c;
      ++start_[size_t(c) + 1];
    }
    for (size_t c = 1; c < start_.size(); ++c)
      start_[c] += start_[c - 1];
    index_.resize(start_.back());
    std::vector<size_t> fill(start_.begin(), start_.end() - 1);
    for (size_t k = begin; k < end;
         ++k) { // counting sort: ascending k inside a cell
      if (cellOf[k - begin] >= 0)
        index_[fill[size_t(cellOf[k - begin])]++] = k;
    }
  }

  /// \brief Call f(k) for every bucketed keypoint whose cell overlaps the
  /// square
  ///        [p - radius, p + radius]; a superset of those within `radius` of p.
  template <class F>
  void forEachNear(const Eigen::Vector2d &p, double radius, F &&f) const {
    const int u0 = clampU(int(std::floor((p[0] - radius) / cell_)));
    const int u1 = clampU(int(std::floor((p[0] + radius) / cell_)));
    const int v0 = clampV(int(std::floor((p[1] - radius) / cell_)));
    const int v1 = clampV(int(std::floor((p[1] + radius) / cell_)));
    for (int v = v0; v <= v1; ++v) {
      for (int u = u0; u <= u1; ++u) {
        const size_t c = size_t(v * nu_ + u);
        for (size_t i = start_[c]; i < start_[c + 1]; ++i)
          f(index_[i]);
      }
    }
  }

private:
  int clampU(int u) const { return std::min(std::max(u, 0), nu_ - 1); }
  int clampV(int v) const { return std::min(std::max(v, 0), nv_ - 1); }
  int cellIndex(double u, double v) const {
    return clampV(int(std::floor(v / cell_))) * nu_ +
           clampU(int(std::floor(u / cell_)));
  }

  double cell_;
  int nu_, nv_;
  std::vector<size_t> start_; ///< CSR offsets per cell (size nu*nv + 1).
  std::vector<size_t> index_; ///< keypoint indices grouped by cell.
};

} // namespace okvis

#endif // INCLUDE_OKVIS_KEYPOINTGRID_HPP_
