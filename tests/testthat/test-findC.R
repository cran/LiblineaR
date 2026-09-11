# findC=TRUE: automatic C search. Upstream LIBLINEAR (and the vendored
# setup_params() in src/trainLinear.c) restricts this to L2R_LR (type 0) and
# L2R_L2LOSS_SVC (type 2); every other type raises a clean R error via
# Rf_error, not a silent no-op.

test_that("findC=TRUE returns a positive numeric C for the supported types", {
  f <- fixture_classif()
  for (tt in c(0, 2)) {
    m <- LiblineaR(f$x, f$y, type = tt, findC = TRUE, cross = 3)
    expect_true(is.numeric(m) && length(m) == 1 && m > 0, info = paste("type", tt))
  }
})

test_that("findC=TRUE on an unsupported type raises a clear R error", {
  f <- fixture_classif()
  # setup_params() (src/trainLinear.c) raises this via Rf_error, a real
  # R-level error rather than a silent failure.
  expect_error(
    LiblineaR(f$x, f$y, type = 1, findC = TRUE, cross = 3),
    "Warm-start parameter search only available"
  )
})

test_that("useInitC=FALSE still runs for a supported type", {
  f <- fixture_classif()
  expect_error(LiblineaR(f$x, f$y, type = 0, findC = TRUE, cross = 3, useInitC = FALSE), NA)
})
