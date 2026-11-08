# Fixture generator for tests/testthat/test-reference-coefficients.R.
#
# NOT a test file: testthat never auto-runs anything under fixtures/. Run it
# deliberately, by hand, from the package root:
#
#   Rscript tests/testthat/fixtures/generate-reference-coefficients.R
#
# This fixture predates the vendor-upgrade golden battery (tests/testthat/
# fixtures/generate-vendor-golden-*.R) and, until now, had no committed
# generator -- the logic below is reverse-engineered from
# test-reference-coefficients.R's own generation loop (fixture_classif()/
# fixture_regr(), type 0-7 and 11-13, cost=1, seed 42 immediately before
# each call) so it can now be regenerated the same deliberate,
# documented-reason way as the rest of the battery, rather than by hand.
#
# LIBLINEAR ships no independent reference test harness to diff against, so
# there is no external "known-good" LIBLINEAR CLI output to pin against
# here -- this freezes coefficients from a verified build as the
# regression baseline, not externally-audited ground truth (see the note
# already in test-reference-coefficients.R).
#
# Regenerate deliberately, never silently, only when a reviewed change to
# the solver is meant to alter these values -- e.g. this was regenerated
# for the LIBLINEAR vendor-upgrade Stage (i) (v2.1 -> v2.30), which merges
# in a real algorithm change (a preconditioned trust-region CG solver,
# upstream v2.21) affecting the primal Newton solvers (types 0, 2, 11);
# confirmed via a direct diff against the pre-upgrade fixture that only
# types 0, 2, 11 moved (max|diff| 4.5e-3, 2.7e-2, 3.7e-4 respectively) and
# every dual/L1 type (1, 3, 4, 5, 6, 7, 12, 13) stayed bit-for-bit
# identical -- see VENDOR-UPGRADE-PLAN.md.

library(LiblineaR)

pkg_root <- normalizePath(".")
if (!file.exists(file.path(pkg_root, "DESCRIPTION"))) {
  stop("Run this script with the package root as the working directory.")
}
source(file.path(pkg_root, "tests", "testthat", "helper-fixtures.R"))

fc <- fixture_classif()
fr <- fixture_regr()

ref <- list()
for (t in 0:7) {
  set.seed(42)
  m <- LiblineaR(fc$x, fc$y, type = t, cost = 1)
  ref[[paste0("type", t)]] <- unname(m$W)
}
for (t in 11:13) {
  set.seed(42)
  m <- suppressWarnings(LiblineaR(fr$x, fr$y, type = t, cost = 1))
  ref[[paste0("type", t)]] <- unname(m$W)
}

out_path <- file.path(pkg_root, "tests", "testthat", "fixtures", "reference-coefficients.rds")
saveRDS(ref, out_path, version = 2)
cat("Wrote", length(ref), "golden coefficient matrices to", out_path, "\n")
