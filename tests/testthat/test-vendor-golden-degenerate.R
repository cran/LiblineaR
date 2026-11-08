# Golden-value regression battery for the vendored LIBLINEAR solver, ahead
# of the planned v2.1 -> v2.50 vendor-upgrade program (src/linear.cpp,
# src/tron.cpp -> newton.cpp, trainLinear.c, predictLinear.c). Complements
# test-vendor-golden-classification.R (breadth across cost/bias/epsilon on
# ordinary, well-conditioned data): this file targets numerically
# stressful/degenerate regimes -- the scenarios most likely to reveal a
# subtle solver-rewrite difference even when "normal" cases pass fine,
# because they sit near a convergence boundary or use a degenerate design
# matrix. See tests/testthat/fixtures/generate-vendor-golden-degenerate.R
# for the full scenario rationale and how to (deliberately) regenerate the
# fixture.
#
# Scenarios: near-perfectly-separable 2-class (large margin), heavily
# overlapping/noisy 2-class, high-dim low-sample-count (p=50, n=15,
# classification + regression), single-feature (p=1, classification +
# regression), duplicated/collinear columns, and the same signal cut into
# 2 classes vs 3 classes. Classification types {0,1,2,3,5,6,7} are pinned
# per scenario (plus type 4, Crammer-Singer, for the 3-class boundary
# case); regression types {11,12,13} are pinned for the high-dim and
# single-feature scenarios.
#
# Reproducibility: re-running this exact grid against the current solver
# reproduces every one of the 56 frozen W matrices bit-for-bit (observed
# max|W_now - W_golden| == 0 across the whole grid, checked directly by
# running the generator twice and diffing, before picking a tolerance).
# tolerance=1e-6 below is therefore generous headroom for build/BLAS-level
# float noise, not a threshold tuned to just barely pass -- any real drift
# (e.g. a legitimately different converged value from a rewritten Newton
# solver) will be many orders of magnitude above it.
#
# Dual solvers (types 1, 3, 4, 7, 12, 13) route randomness through R's own
# RNG (rand_between() in src/linear.cpp), so set.seed() is reset
# immediately before every single LiblineaR() call below, matching how the
# fixture itself was generated.
#
# Datasets are loaded from the frozen fixture (not regenerated here) so
# this test can never silently drift from the exact data the golden values
# were pinned against; only the training + comparison is "fresh".

GOLDEN_TOL <- 1e-6
TRAINING_SEED <- 777

golden <- readRDS(test_path("fixtures", "vendor-golden-degenerate.rds"))
ds <- golden$datasets

CLF_TYPES  <- c(0, 1, 2, 3, 5, 6, 7)
CS_TYPES   <- c(0, 1, 2, 3, 4, 5, 6, 7)
REGR_TYPES <- c(11, 12, 13)

# Trains one type on (x, y), checks it isn't garbage (no NA/Inf, correct
# shape) BEFORE comparing to the frozen golden value, so a real
# crash/garbage-output regression fails loudly and specifically instead of
# just diffing NA against a number; then compares W to the golden fixture.
# expect_nrow is 1 for every binary-classification and regression scenario,
# and nlevels(y) for the 3-class boundary scenario.
check_type <- function(x, y, type, key, expect_nrow) {
  set.seed(TRAINING_SEED)
  m <- LiblineaR(x, y, type = type, cost = 1, svr_eps = 0.01)
  w <- unname(m$W)

  expect_false(any(is.na(w)), label = paste(key, "no NA"))
  expect_true(all(is.finite(w)), label = paste(key, "all finite"))
  expect_equal(nrow(w), expect_nrow, label = paste(key, "nrow(W)"))
  expect_equal(ncol(w), ncol(x) + 1, label = paste(key, "ncol(W)"))

  expect_equal(w, golden$results[[key]], tolerance = GOLDEN_TOL, label = key)
}

# Runs one scenario's full type grid and returns how many types were
# checked (used only as a grid-coverage sanity check below).
run_scenario <- function(scenario, x, y, types, expect_nrow) {
  for (t in types) {
    key <- paste0(scenario, ".type", t)
    check_type(x, y, t, key, expect_nrow)
  }
  length(types)
}

test_that("near-perfectly-separable 2-class data golden grid", {
  # Risk: huge margin -> gradients vanish fast near the optimum, so a
  # rewritten stopping criterion (e.g. the v2.40 Newton solver replacing
  # TRON's trust-region stopping rule) is most likely to under- or
  # over-iterate exactly here, before it shows up on harder data.
  f <- ds$near_separable
  n <- run_scenario("near_separable", f$x, f$y, CLF_TYPES, 1)
  expect_equal(n, 7)
})

test_that("heavily overlapping/noisy 2-class data golden grid", {
  # Risk: no clean margin means no single dominant solution, so a
  # solver-internal change to iteration order or tie-breaking (rather than
  # to the underlying math) is more likely to visibly move W here than on
  # cleanly separable data.
  f <- ds$overlap
  n <- run_scenario("overlap", f$x, f$y, CLF_TYPES, 1)
  expect_equal(n, 7)
})

test_that("high-dimensional low-sample-count (p=50, n=15) classification golden grid", {
  # Risk: p >> n inverts which of primal (iterates over p) vs dual
  # (iterates over n) solvers does more work per pass; L1 types 5/6 must
  # also select a sparse subset of 50 features from only 15 rows, a
  # regime the ordinary n >> p fixtures never exercise.
  f <- ds$highdim
  n <- run_scenario("highdim_clf", f$x, f$y_clf, CLF_TYPES, 1)
  expect_equal(n, 7)
})

test_that("high-dimensional low-sample-count (p=50, n=15) regression golden grid", {
  # Risk: same p >> n stress as the classification case above, but on the
  # SVR primal (11) and dual (12, 13) solvers, which use a different
  # loss/optimizer family than classification entirely.
  f <- ds$highdim
  n <- run_scenario("highdim_regr", f$x, f$y_regr, REGR_TYPES, 1)
  expect_equal(n, 3)
})

test_that("single-feature (p=1) classification golden grid", {
  # Risk: W has a single coefficient column, so a per-feature
  # indexing/stride bug that only manifests when p>1 (e.g. an
  # off-by-one in a loop bound touched by the vendor upgrade) cannot mask
  # itself behind other features here.
  f <- ds$single_feat
  n <- run_scenario("single_feat_clf", f$x, f$y_clf, CLF_TYPES, 1)
  expect_equal(n, 7)
})

test_that("single-feature (p=1) regression golden grid", {
  # Risk: same single-column stress as the classification case above, on
  # the SVR solvers instead.
  f <- ds$single_feat
  n <- run_scenario("single_feat_regr", f$x, f$y_regr, REGR_TYPES, 1)
  expect_equal(n, 3)
})

test_that("duplicated/collinear-column data golden grid", {
  # Risk: x3 is an exact linear combination of x1 and x2, so the design
  # matrix is rank-deficient; a numerically fragile Hessian/Gram-matrix
  # implementation change in the vendor upgrade is more likely to show
  # measurable drift here than on a full-rank design.
  f <- ds$collinear
  n <- run_scenario("collinear", f$x, f$y, CLF_TYPES, 1)
  expect_equal(n, 7)
})

test_that("same signal cut into 2 classes golden grid (multiclass-boundary pair, 2-class half)", {
  # Risk: paired with the 3-class test below on the IDENTICAL underlying
  # signal, so any difference between the two tests' drift patterns after
  # a vendor-upgrade stage isolates a multiclass-dispatch-specific
  # regression from one that also affects plain binary classification.
  f <- ds$boundary
  n <- run_scenario("boundary2", f$x, f$y2, CLF_TYPES, 1)
  expect_equal(n, 7)
})

test_that("same signal cut into 3 classes golden grid (multiclass-boundary pair, 3-class half, incl. Crammer-Singer)", {
  # Risk: same underlying signal as the 2-class test above, now dispatched
  # through genuine multiclass one-vs-rest AND, via type 4, the entirely
  # separate Crammer-Singer multiclass solver -- a regression confined to
  # multiclass dispatch will show here and not in the 2-class pair test.
  f <- ds$boundary
  n <- run_scenario("boundary3", f$x, f$y3, CS_TYPES, 3)
  expect_equal(n, 8)
})
