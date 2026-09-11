# The `cross` argument: k-fold cross-validation accuracy (classification) or
# mean-squared error (regression), returned as a single number rather than a
# model object.

test_that("cross=k on dense classification data returns a plausible, reproducible accuracy", {
  f <- fixture_classif()
  set.seed(55); acc1 <- LiblineaR(f$x, f$y, type = 0, cross = 5)
  set.seed(55); acc2 <- LiblineaR(f$x, f$y, type = 0, cross = 5)

  expect_true(is.numeric(acc1) && length(acc1) == 1)
  expect_true(acc1 >= 0 && acc1 <= 1)
  expect_identical(acc1, acc2)
})

test_that("cross=k on regression data returns a plausible mean-squared error", {
  f <- fixture_regr()
  m <- LiblineaR(f$x, f$y, type = 11, cross = 5, svr_eps = 0.01)
  expect_true(is.numeric(m) && length(m) == 1)
  expect_true(m >= 0)
})

test_that("cross rejects out-of-range values", {
  f <- fixture_classif()
  expect_error(LiblineaR(f$x, f$y, type = 0, cross = -1), "cannot be negative")
  expect_error(LiblineaR(f$x, f$y, type = 0, cross = nrow(f$x) + 1), "cannot be larger")
})
