# CORE (vendor-upgrade safety net): pins exact W coefficients for the three
# regression solvers (type 11 = L2R_L2LOSS_SVR primal, type 12 =
# L2R_L2LOSS_SVR_DUAL, type 13 = L2R_L1LOSS_SVR_DUAL) across a real
# parameter grid, from the CURRENT, already verified-correct vendored
# LIBLINEAR solver, ahead of the multi-stage vendor-upgrade to v2.50.
#
# This complements, and does not duplicate, existing regression coverage:
# test-reference-coefficients.R pins types 11-13 at cost=1 only, with no
# bias/svr_eps variation and a single dataset; test-solver-equivalence.R
# checks primal/dual agreement (types 11 vs 12) with no golden pinning at
# all. Here the grid is type x cost x bias x svr_eps x dataset, so a
# vendor-upgrade stage that only shifts behaviour at an extreme cost, only
# with a bias term on/off, only at a tight/loose svr_eps, or only on a
# noisier/nonlinear-ish dataset cannot hide behind a single-point check.
#
# Provenance of tests/testthat/fixtures/vendor-golden-svr-paramgrid.rds: see
# tests/testthat/fixtures/generate-vendor-golden-svr-paramgrid.R (the
# generator that produced it, runnable via
#   Rscript tests/testthat/fixtures/generate-vendor-golden-svr-paramgrid.R
# ). LIBLINEAR ships no independent reference harness to diff against, so
# these are a regression baseline frozen from a verified build, not
# externally-audited ground truth. Regenerate deliberately (never
# silently, never automatically) only when a REVIEWED vendor-upgrade stage
# is meant to alter these values (e.g. the v2.40 Newton-solver rewrite
# legitimately moving converged values within tolerance) -- any other
# change here after an upgrade stage is a candidate regression.
#
# Dual solvers (types 12, 13) route randomness through R's own RNG
# (rand_between() in src/linear.cpp), so set.seed() is called immediately
# before EVERY LiblineaR() call below (type 11 is primal and doesn't
# strictly need it, but is seeded too for uniformity with the generator
# script's loop) -- not once at the top, so results stay reproducible.
#
# Tolerance: checked directly against the actual observed
# max(abs(fresh - frozen)) across this whole grid before picking a value
# (see the comment above the loop) rather than guessing a round number.

# Second regression dataset, duplicated verbatim from
# generate-vendor-golden-svr-paramgrid.R (kept local here rather than
# sourced, so this test file has no dependency on the generator script at
# test time). Higher noise relative to signal, and a nonlinear-ish target
# (quadratic + interaction terms) fit with a linear model, so residuals --
# and therefore svr_eps's insensitive-tube effect -- are non-trivial,
# unlike fixture_regr()'s near-perfectly-linear, near-noiseless target.
# n=50, 4 features, own explicit seed (777), independent of
# fixture_regr()'s internal seed (321).
fixture_regr_highnoise <- function() {
  set.seed(777)
  n <- 50
  x <- matrix(rnorm(n * 4, sd = 2), ncol = 4)
  colnames(x) <- c("x1", "x2", "x3", "x4")
  y <- as.vector(
    0.8 * x[, 1] - 0.3 * x[, 2]^2 + 0.5 * x[, 1] * x[, 3] + 0.2 * x[, 4]
  ) + rnorm(n, sd = 3)
  list(x = x, y = y)
}

test_that("SVR types 11-13 match frozen golden W across the cost x bias x svr_eps x dataset grid", {
  # Risk covered: any solver-code change (src/linear.cpp's
  # solve_l2r_l2loss_svr_primal/dual or solve_l2r_l1l2_svr, or
  # src/tron.cpp's Newton/CG iteration that the type-11 primal path relies
  # on) that shifts converged coefficients for ANY point in this grid --
  # not just the single (cost=1, bias=1, svr_eps=default) point that
  # test-reference-coefficients.R covers. Extreme cost (0.01, 100)
  # exercises numerically stiff regularization; bias on/off exercises the
  # extra intercept column; tight/loose svr_eps (0.001 vs 1) exercises the
  # insensitive-tube width that only matters once residuals are
  # non-trivial, which is why a second, noisier/nonlinear-ish dataset is
  # included alongside the easy fixture_regr() one.
  golden <- readRDS(test_path("fixtures", "vendor-golden-svr-paramgrid.rds"))

  datasets <- list(
    regr    = fixture_regr(),
    hinoise = fixture_regr_highnoise()
  )

  types    <- c(11, 12, 13)
  costs    <- c(0.01, 1, 100)
  biases   <- c(-1, 1)
  svr_epss <- c(0.001, 0.1, 1)

  # Tolerance set from the actual observed max(abs(fresh - frozen)) over
  # the whole grid (checked directly, not guessed): re-running the exact
  # same grid through the exact same solver build is bit-reproducible to
  # within ordinary floating-point noise, so 1e-8 is tight enough to catch
  # a real numeric drift while not being brittle to compiler/BLAS-level
  # noise across runs on the same machine.
  tol <- 1e-8

  for (ds_name in names(datasets)) {
    ds <- datasets[[ds_name]]
    for (t in types) {
      for (co in costs) {
        for (bi in biases) {
          for (ep in svr_epss) {
            key <- paste0(
              "type", t,
              "_cost", co,
              "_bias", bi,
              "_eps", ep,
              "_", ds_name
            )
            set.seed(42)
            m <- suppressWarnings(
              LiblineaR(ds$x, ds$y, type = t, cost = co, bias = bi, svr_eps = ep)
            )
            expect_equal(
              unname(m$W), golden[[key]],
              tolerance = tol, label = key
            )
          }
        }
      }
    }
  }
})
