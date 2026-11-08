# Generator for tests/testthat/fixtures/vendor-golden-svr-paramgrid.rds
#
# Plain R script -- NOT a test file, never auto-run by testthat. Run it
# deliberately, from the package root, with:
#   Rscript tests/testthat/fixtures/generate-vendor-golden-svr-paramgrid.R
#
# WHAT IT CAPTURES: this is the golden-value safety net for the regression
# solvers (type 11 = L2R_L2LOSS_SVR primal, type 12 = L2R_L2LOSS_SVR_DUAL,
# type 13 = L2R_L1LOSS_SVR_DUAL), pinned from the CURRENT, already
# verified-correct vendored LIBLINEAR solver (~v2.1) ahead of the
# multi-stage vendor-upgrade to v2.50 (touching src/linear.cpp and
# src/tron.cpp -> newton.cpp in particular). Unlike
# test-reference-coefficients.R (which pins these 3 types at cost=1 only,
# no bias/svr_eps variation), this freezes a full parameter-grid crossing:
#   - type       in {11, 12, 13}
#   - cost       in {0.01, 1, 100}
#   - bias       in {-1 (off), 1}
#   - svr_eps    in {0.001, 0.1, 1}
#   - dataset    in {fixture_regr(), fixture_regr_highnoise() -- the
#                 latter defined in this script, see below}
# i.e. the full 3 x 3 x 2 x 3 x 2 = 108-point cross-product, so a
# vendor-upgrade stage that shifts behaviour only at extreme cost, only
# with a bias term, only at tight/loose svr_eps, or only on a noisier/
# nonlinear-ish dataset cannot hide behind a single-point check.
#
# fixture_regr_highnoise(): fixture_regr() (from helper-fixtures.R) is a
# small, nearly-noiseless linear dataset (signal sd >> noise sd = 0.1) --
# deliberately easy, so residuals are close to zero and don't exercise the
# solvers' handling of a harder fit. This second dataset uses a different
# regime: more samples (n=50), more features (4), a nonlinear-ish target
# (quadratic + interaction term) that is still fit with a linear model,
# and noise on the same order as the signal -- so residuals, and hence the
# svr_eps insensitive-tube behaviour, are non-trivial. It carries its own
# explicit set.seed() (777), independent of fixture_regr()'s internal
# seed (321), and is duplicated verbatim (not sourced) in
# test-vendor-golden-svr-paramgrid.R so the test file can regenerate the
# same data without depending on this generator script.
#
# These are regression-baseline values, not externally-audited ground
# truth (LIBLINEAR ships no independent reference harness to diff
# against). Re-run this script deliberately (never silently, never as
# part of an automated test run) via
#   Rscript tests/testthat/fixtures/generate-vendor-golden-svr-paramgrid.R
# only when a REVIEWED vendor-upgrade stage is meant to change these
# values (e.g. the v2.40 Newton-solver rewrite legitimately moving
# converged values within tolerance). Any other change in these numbers
# after a vendor-upgrade stage is a candidate regression, not a reason to
# regenerate.
#
# Dual solvers (types 12, 13) route randomness through R's own RNG
# (rand_between() in src/linear.cpp), so set.seed() is called immediately
# before EVERY LiblineaR() call below, not once at the top of the script
# (type 11 is primal and doesn't need it, but is seeded too for
# uniformity and so the loop body doesn't need a type-conditional) --
# matching the corresponding test file's discipline so results are
# reproducible run to run.

library(LiblineaR)

# This script is run directly with Rscript (cwd = package root), so
# source the fixture helper directly rather than relying on testthat's
# auto-loading of helper-*.R (which only happens inside a testthat run).
source(file.path("tests", "testthat", "helper-fixtures.R"))

# Second regression dataset: higher noise relative to signal, and a
# nonlinear-ish target (quadratic + interaction terms) fit with a linear
# model, so residuals -- and therefore svr_eps's insensitive-tube effect
# -- are non-trivial, unlike fixture_regr()'s near-perfectly-linear,
# near-noiseless target. n=50, 4 features. Own explicit seed, kept local
# to this script (and duplicated verbatim in the test file).
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

datasets <- list(
  regr    = fixture_regr(),
  hinoise = fixture_regr_highnoise()
)

types    <- c(11, 12, 13)
costs    <- c(0.01, 1, 100)
biases   <- c(-1, 1)
svr_epss <- c(0.001, 0.1, 1)

golden <- list()

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
          golden[[key]] <- unname(m$W)
        }
      }
    }
  }
}

out_path <- file.path(
  "tests", "testthat", "fixtures", "vendor-golden-svr-paramgrid.rds"
)
saveRDS(golden, out_path, version = 2)
cat("Wrote", out_path, "\n")
cat(length(golden), "golden W matrices\n")
