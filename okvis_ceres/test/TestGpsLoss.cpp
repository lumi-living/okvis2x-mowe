/**
 * mow-e (T-0146, ADR-0042 §(2)): the GNSS robust loss is selected by GpsParameters
 * (gps_parameters.loss / loss_scale) and weighs a fix as the ADR update's table says.
 * The GNSS residual is whitened, so a fix d metres off at sigma σ has s = (d/σ)² on one
 * axis; the IRLS weight is rho'(s) and the pull (influence) is rho'(s)·d/σ [σ units].
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */
#include <algorithm>
#include <cmath>
#include <ceres/loss_function.h>
#include <gtest/gtest.h>
#include <okvis/ViGraph.hpp>
#include <okvis/Parameters.hpp>

namespace {

double weight(const ::ceres::LossFunction* loss, double d_m, double sigma_m) {
  double rho[3];
  const double r = d_m / sigma_m;
  loss->Evaluate(r * r, rho);
  return rho[1];
}

const ::ceres::LossFunction* lossFor(okvis::ViGraph& graph, const std::string& name, double scale) {
  okvis::GpsParameters p;
  p.type = "cartesian";
  if (!name.empty()) { p.loss = name; p.lossScale = scale; }
  graph.addGps(p);
  return graph.gpsLossFunction();
}

}  // namespace

TEST(GpsLoss, DefaultIsUpstreamCauchy3) {
  okvis::GpsParameters p;
  EXPECT_EQ(p.loss, "cauchy");
  EXPECT_DOUBLE_EQ(p.lossScale, 3.0);
  okvis::ViGraph graph;
  const ::ceres::LossFunction* loss = lossFor(graph, "", 0.0);
  ASSERT_NE(loss, nullptr);
  EXPECT_NEAR(weight(loss, 0.03, 0.01), 0.5, 1e-12);  // Cauchy(a): rho' = 1 / (1 + s/a²), 3σ -> 0.5
}

TEST(GpsLoss, WeightTableAtRtkSigma) {
  const double sigma = 0.014;  // RTK FIXED hAcc on the bench receiver (T-0144)
  okvis::ViGraph cauchyGraph, huberGraph;
  const ::ceres::LossFunction* cauchy = lossFor(cauchyGraph, "cauchy", 3.0);
  const ::ceres::LossFunction* huber = lossFor(huberGraph, "huber", 10.0);
  // Cauchy(3σ): a 4 cm excursion halves a fix, 10 cm leaves 0.15, 1 m ~ 1.6e-3.
  EXPECT_NEAR(weight(cauchy, 0.04, sigma), 1.0 / (1.0 + std::pow(0.04 / sigma / 3.0, 2)), 1e-12);
  EXPECT_NEAR(weight(cauchy, 0.04, sigma), 0.52, 0.01);
  EXPECT_NEAR(weight(cauchy, 0.10, sigma), 0.15, 0.01);
  EXPECT_NEAR(weight(cauchy, 1.00, sigma), 1.8e-3, 1e-4);
  // Huber(10σ): full weight up to 14 cm, then k/r -- a 1 m fix pulls like a 14 cm one.
  EXPECT_DOUBLE_EQ(weight(huber, 0.04, sigma), 1.0);
  EXPECT_DOUBLE_EQ(weight(huber, 0.10, sigma), 1.0);
  EXPECT_NEAR(weight(huber, 1.00, sigma), 10.0 * sigma / 1.0, 1e-12);
  EXPECT_NEAR(weight(huber, 1.00, sigma) * 1.00 / sigma, 10.0, 1e-9);   // influence capped at 10σ
  // the point of the change: past ~3σ Cauchy's pull falls with distance, Huber's never does
  for (double d : {0.1, 0.3, 1.0, 3.0}) {
    EXPECT_GT(weight(huber, d, sigma), 5.0 * weight(cauchy, d, sigma)) << d;
    EXPECT_GE(weight(huber, d, sigma) * d / sigma, std::min(d / sigma, 10.0) - 1e-9) << d;  // r below the knee, 10σ above
    EXPECT_LT(weight(cauchy, d, sigma) * d / sigma, 1.5) << d;   // Cauchy(3) influence peaks at 1.5σ at r=3
  }
}

TEST(GpsLoss, HuberKneeFollowsLossScale) {
  okvis::ViGraph graph;
  const ::ceres::LossFunction* loss = lossFor(graph, "huber", 30.0);
  EXPECT_DOUBLE_EQ(weight(loss, 0.29, 0.01), 1.0);
  EXPECT_NEAR(weight(loss, 0.60, 0.01), 0.5, 1e-12);
}
