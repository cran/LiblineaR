# expect_error() for the remaining stop() sites in LiblineaR() not already
# covered by test-p0-fixes.R (cost/bias/epsilon/svr_eps) or
# test-cross-validation.R (cross out of range).

test_that("LiblineaR() rejects an unknown model type", {
  f <- fixture_classif()
  expect_error(LiblineaR(f$x, f$y, type = 999), "Unknown model type")
})

test_that("LiblineaR() rejects a target with the wrong dimension", {
  f <- fixture_classif()
  expect_error(LiblineaR(f$x, matrix(f$y, nrow = 2)), "Wrong dimension for target")
})

test_that("LiblineaR() rejects a target length that disagrees with nrow(data)", {
  f <- fixture_classif()
  expect_error(LiblineaR(f$x, f$y[1:5]), "disagrees")
})

test_that("LiblineaR() rejects a single-class target", {
  f <- fixture_classif()
  expect_error(LiblineaR(f$x, factor(rep("a", nrow(f$x)))), "Wrong number of classes")
})

test_that("LiblineaR() rejects an unnamed wi", {
  f <- fixture_classif()
  expect_error(LiblineaR(f$x, f$y, wi = c(1, 2)), "wi has to be a named vector")
})

test_that("LiblineaR() rejects a non-integer numeric classification target instead of silently truncating it", {
  f <- fixture_classif()
  n <- nrow(f$x)
  # Two distinct values that truncate (as.integer()) to the same integer --
  # previously silently collapsed into a single class with no error.
  y_frac <- rep(c(1.1, 1.9), length.out = n)
  expect_error(LiblineaR(f$x, y_frac, type = 0), "non-integer")

  # Whole-number doubles must still be accepted (e.g. 0/1 targets, common
  # when a caller passes an as.numeric()-coerced logical or factor code).
  y_whole <- rep(c(0, 1), length.out = n)
  expect_error(LiblineaR(f$x, y_whole, type = 0), NA)
})

test_that("predict() rejects an object that isn't a fitted LiblineaR model", {
  # predict.LiblineaR() isn't exported (it's registered as an S3 method), so
  # it's called directly here to test its own object-class guard regardless
  # of what predict()'s generic dispatch would otherwise do with a
  # non-"LiblineaR" object.
  predict_LiblineaR <- getS3method("predict", "LiblineaR")
  f <- fixture_classif()
  expect_error(predict_LiblineaR(list(W = matrix(1, 1, 3)), f$x), "class 'LiblineaR'")
  expect_error(predict_LiblineaR(lm(rnorm(5) ~ 1), f$x), "class 'LiblineaR'")

  # A genuine model is unaffected by the new guard.
  m <- LiblineaR(f$x, f$y, type = 0)
  expect_error(predict_LiblineaR(m, f$x), NA)
})

test_that("NA or Inf in data errors cleanly rather than silently training", {
  # R's own .C() interface rejects NA/NaN/Inf arguments by default (NAOK=FALSE),
  # so this is already safe; asserted here so it stays that way.
  f <- fixture_classif()
  x_na <- f$x; x_na[1, 1] <- NA
  x_inf <- f$x; x_inf[1, 1] <- Inf
  expect_error(LiblineaR(x_na, f$y, type = 0), "NA/NaN/Inf")
  expect_error(LiblineaR(x_inf, f$y, type = 0), "NA/NaN/Inf")
})
