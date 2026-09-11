# CORE: the primary defence against a silent regression in the solver code
# (src/linear.cpp, src/tron.cpp, or the trainLinear.c/predictLinear.c glue).
#
# Provenance of tests/testthat/fixtures/reference-coefficients.rds: LIBLINEAR
# ships no independent reference test harness to diff against, so there is
# no external "known-good" LIBLINEAR CLI output to pin against here. This
# fixture freezes coefficients from a verified build as the regression
# baseline, not externally-audited ground truth. Regenerate deliberately
# (never silently) via the seed/fixture setup below if a reviewed change to
# the solver is meant to alter these values.

test_that("classification and regression coefficients match the frozen reference, per type", {
  ref <- readRDS(test_path("fixtures", "reference-coefficients.rds"))
  fc <- fixture_classif()
  fr <- fixture_regr()

  for (t in 0:7) {
    set.seed(42)
    m <- LiblineaR(fc$x, fc$y, type = t, cost = 1)
    expect_equal(unname(m$W), ref[[paste0("type", t)]], tolerance = 1e-6, label = paste("type", t))
  }

  for (t in 11:13) {
    set.seed(42)
    m <- suppressWarnings(LiblineaR(fr$x, fr$y, type = t, cost = 1))
    expect_equal(unname(m$W), ref[[paste0("type", t)]], tolerance = 1e-6, label = paste("type", t))
  }
})
