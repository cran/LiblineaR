# VENDOR-UPGRADE GOLDEN BATTERY (Stage (ii) addition): near-separable data
# CROSSED WITH tight epsilon, for the 3 types the tron.cpp -> newton.cpp
# rewrite actually touches (0 = L2R_LR primal, 2 = L2R_L2LOSS_SVC primal,
# 11 = L2R_L2LOSS_SVR primal).
#
# WHY THIS FILE EXISTS (gap it closes)
# -------------------------------------
# Stage (ii) (v2.30 -> v2.40) replaced src/tron.cpp's trust-region Newton
# solver with src/newton.cpp's Armijo line-search Newton-CG solver, for
# types 0/2/11 only (every dual/L1 type is confirmed bit-for-bit unaffected
# by this stage). A stopping-criterion difference between the two
# algorithms is most likely to surface exactly where the gradient is
# already close to flat AND the solver is asked to keep pushing past where
# a looser criterion would have stopped -- i.e. near-separable data (tiny
# gradient near the optimum) evaluated at a TIGHT epsilon (forces deep
# iteration into that flat region). No existing fixture in this battery
# tests both axes at once for these types:
#   - test-vendor-golden-degenerate.R's near_separable scenario always uses
#     the solver's default epsilon (never tightened).
#   - test-vendor-golden-classweight-epsilon-bias.R's tightest epsilon
#     (1e-6) is exercised only on an ordinary 90/10-imbalanced dataset that
#     is NOT near-separable, and excludes type 11 (regression) entirely --
#     it is a classification-only fixture.
#
# This file's golden values are frozen in
# tests/testthat/fixtures/vendor-golden-newton-stress.rds, produced by
# tests/testthat/fixtures/generate-vendor-golden-newton-stress.R (see that
# script's header for full scenario rationale and the deliberate,
# never-silent regeneration policy -- the same policy as every other
# vendor-golden-*.rds fixture in this project).
#
# Tolerance: re-running this exact grid against the current solver
# reproduced every one of the 12 frozen W matrices bit-for-bit (observed
# max|W_now - W_golden| == 0 across the whole grid, checked directly before
# writing this file, the same way every other vendor-golden-*.R file in
# this battery picked its tolerance). 1e-6 below is therefore generous
# headroom for build/BLAS-level float noise, not a threshold tuned to just
# barely pass.
#
# Types 0, 2 and 11 are all PRIMAL solvers and do not route randomness
# through R's RNG (rand_between() in src/linear.cpp is a DUAL-solver-only
# code path) -- but set.seed() is still reset immediately before every
# LiblineaR() call below, matching the convention used uniformly elsewhere
# in this golden battery, so this file's pattern stays identical to the
# others' regardless of type.
#
# The near-separable classification dataset and the near-deterministic
# regression dataset are loaded from the frozen fixture (not regenerated
# here) so this test can never silently drift from the exact data the
# golden values were pinned against; only the training + comparison is
# "fresh".

GOLDEN_TOL <- 1e-6
TRAINING_SEED <- 777

golden <- readRDS(test_path("fixtures", "vendor-golden-newton-stress.rds"))
ds <- golden$datasets
ns <- ds$near_separable_stress
nd <- ds$near_deterministic_regr

EPSILONS <- c(1e-6, 1e-8)
COSTS <- c(1, 100)
SVR_EPS <- 0.001

# Trains one type/epsilon/cost combination, checks the result isn't garbage
# (no NA/Inf, correct shape) BEFORE comparing to the frozen golden value --
# so a real crash/garbage-output regression (e.g. the Newton rewrite
# diverging instead of converging in this flat-gradient regime) fails
# loudly and specifically, rather than just diffing NA against a number --
# then compares W to the golden fixture.
check_type <- function(x, y, type, eps, cost, key, ...) {
  set.seed(TRAINING_SEED)
  m <- LiblineaR(x, y, type = type, cost = cost, epsilon = eps, ...)
  w <- unname(m$W)

  expect_false(any(is.na(w)), label = paste(key, "no NA"))
  expect_true(all(is.finite(w)), label = paste(key, "all finite"))
  expect_equal(nrow(w), 1, label = paste(key, "nrow(W)"))
  expect_equal(ncol(w), ncol(x) + 1, label = paste(key, "ncol(W)"))

  expect_equal(w, golden$results[[key]], tolerance = GOLDEN_TOL, label = key)
}

test_that("near-separable classification, type 0 (L2R_LR primal), tight epsilon x cost golden grid", {
  # Risk: this type's primal solve is exactly the code path the
  # tron.cpp -> newton.cpp rewrite replaced (Newton-CG w/ Armijo line
  # search, was trust-region Newton). Near-separable data pushes the
  # gradient toward zero fast; a tight epsilon then forces the solver to
  # keep iterating deep into that flat region, exactly where the two
  # algorithms' stopping rules are most likely to disagree.
  for (eps in EPSILONS) {
    for (cost in COSTS) {
      key <- paste0("type0_eps", format(eps, scientific = TRUE), "_cost", cost)
      check_type(ns$x, ns$y, type = 0, eps = eps, cost = cost, key = key)
    }
  }
})

test_that("near-separable classification, type 2 (L2R_L2LOSS_SVC primal), tight epsilon x cost golden grid", {
  # Risk: same Newton-rewrite exposure as type 0 above, but on the
  # squared-hinge loss instead of logistic loss -- a different
  # objective/gradient shape feeding the same solver internals.
  for (eps in EPSILONS) {
    for (cost in COSTS) {
      key <- paste0("type2_eps", format(eps, scientific = TRUE), "_cost", cost)
      check_type(ns$x, ns$y, type = 2, eps = eps, cost = cost, key = key)
    }
  }
})

test_that("near-deterministic regression, type 11 (L2R_L2LOSS_SVR primal), tight epsilon x cost golden grid", {
  # Risk: the third and last Newton-affected type, and the one NO existing
  # fixture in this battery tests at a tight epsilon at all (the
  # classweight-epsilon-bias fixture excludes type 11 entirely). A
  # near-deterministic target (noise sd 0.001, three orders of magnitude
  # below signal) keeps residuals -- and hence the L2-loss gradient --
  # tiny almost everywhere at the optimum, the regression analogue of the
  # classification scenarios' vanishing gradient above.
  for (eps in EPSILONS) {
    for (cost in COSTS) {
      key <- paste0("type11_eps", format(eps, scientific = TRUE), "_cost", cost)
      check_type(nd$x, nd$y, type = 11, eps = eps, cost = cost, key = key,
                  svr_eps = SVR_EPS)
    }
  }
})

test_that("grid coverage sanity: exactly 12 golden values, one per type x epsilon x cost combination", {
  # Not a numeric check -- guards against a future edit silently dropping a
  # combination from the loops above without anyone noticing the battery
  # got thinner.
  expect_equal(length(golden$results), 3 * length(EPSILONS) * length(COSTS))
})
