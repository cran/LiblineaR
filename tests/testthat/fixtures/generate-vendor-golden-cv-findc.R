# Generator for tests/testthat/fixtures/vendor-golden-cv-findc.rds
#
# Plain R script -- NOT a test file, never auto-run by testthat. Run it
# deliberately, from the package root, with:
#   Rscript tests/testthat/fixtures/generate-vendor-golden-cv-findc.R
#
# WHAT IT CAPTURES: this is the golden-value safety net for the
# cross-validation (cross=k) and findC=TRUE dimension of LiblineaR, pinned
# from the CURRENT, already verified-correct vendored LIBLINEAR solver
# (~v2.1) ahead of the multi-stage vendor-upgrade to v2.50. It freezes:
#   - cross=k returned accuracy for classification (fixture_classif(),
#     types 0-7, k in {3,5,10})
#   - cross=k returned MSE for regression (fixture_regr(), types 11-13,
#     k in {3,5,10})
#   - findC=TRUE discovered C, for the currently-supported types (0 and 2
#     only -- see src/trainLinear.c's setup_params(), which raises
#     "Warm-start parameter search only available for L2R_LR and
#     L2R_L2LOSS_SVC" for any other type), crossed with
#     useInitC in {TRUE, FALSE} and starting cost in {1, 1000}, cross in
#     {3,5}
#
# These are regression-baseline values, not externally-audited ground
# truth (LIBLINEAR ships no independent reference harness to diff
# against). Re-run this script deliberately (never silently, never as
# part of an automated test run) via
#   Rscript tests/testthat/fixtures/generate-vendor-golden-cv-findc.R
# only when a REVIEWED vendor-upgrade stage is meant to change these
# values (e.g. the v2.40 Newton-solver rewrite legitimately moving
# converged values within tolerance). Any other change in these numbers
# after a vendor-upgrade stage is a candidate regression, not a reason to
# regenerate.
#
# Dual solvers (types 1,3,4,7,12,13) route randomness for cross-validation
# fold assignment through R's own RNG (rand_between() in src/linear.cpp),
# so set.seed() is called immediately before EVERY LiblineaR() call below,
# not once at the top of the script -- matching the corresponding test
# file's discipline so results are reproducible run to run.

library(LiblineaR)

# This script is run directly with Rscript (cwd = package root), so
# source the fixture helper directly rather than relying on testthat's
# auto-loading of helper-*.R (which only happens inside a testthat run).
source(file.path("tests", "testthat", "helper-fixtures.R"))

fc <- fixture_classif()
fr <- fixture_regr()

golden <- list()

# --- cross=k classification accuracy, types 0-7, k in {3,5,10} ---
golden$cross_classif <- list()
for (t in 0:7) {
  for (k in c(3, 5, 10)) {
    key <- paste0("type", t, "_k", k)
    set.seed(777)
    golden$cross_classif[[key]] <- LiblineaR(fc$x, fc$y, type = t, cross = k)
  }
}

# --- cross=k regression MSE, types 11-13, k in {3,5,10} ---
golden$cross_regr <- list()
for (t in 11:13) {
  for (k in c(3, 5, 10)) {
    key <- paste0("type", t, "_k", k)
    set.seed(777)
    golden$cross_regr[[key]] <- suppressWarnings(
      LiblineaR(fr$x, fr$y, type = t, cross = k, svr_eps = 0.01)
    )
  }
}

# --- findC=TRUE discovered C, types 0 and 2 only, useInitC x cost x cross ---
golden$findC <- list()
for (t in c(0, 2)) {
  for (useInitC in c(TRUE, FALSE)) {
    for (cost0 in c(1, 1000)) {
      for (k in c(3, 5)) {
        key <- paste0(
          "type", t,
          "_useInitC", useInitC,
          "_cost", cost0,
          "_k", k
        )
        set.seed(777)
        golden$findC[[key]] <- LiblineaR(
          fc$x, fc$y,
          type = t, cost = cost0, findC = TRUE,
          useInitC = useInitC, cross = k
        )
      }
    }
  }
}

out_path <- file.path("tests", "testthat", "fixtures", "vendor-golden-cv-findc.rds")
saveRDS(golden, out_path, version = 2)
cat("Wrote", out_path, "\n")
cat("cross_classif:", length(golden$cross_classif), "values\n")
cat("cross_regr:", length(golden$cross_regr), "values\n")
cat("findC:", length(golden$findC), "values\n")
