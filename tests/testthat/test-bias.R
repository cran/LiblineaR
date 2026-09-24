# The bias argument: numeric magnitude and boundary values (0, negative), and
# their effect on W's dimensions. bias=TRUE/FALSE and invalid bias values are
# already covered in test-p0-fixes.R.

test_that("bias>0 adds one column to W; bias<=0 does not", {
  f <- fixture_classif()

  for (b in list(1, 5, 0.01)) {
    m <- LiblineaR(f$x, f$y, type = 0, bias = b)
    expect_equal(dim(m$W), c(1L, ncol(f$x) + 1L), label = paste("bias =", b))
  }
  for (b in list(0, -1, -5)) {
    m <- LiblineaR(f$x, f$y, type = 0, bias = b)
    expect_equal(dim(m$W), c(1L, ncol(f$x)), label = paste("bias =", b))
  }
})

test_that("bias magnitude is actually used, not just its sign", {
  f <- fixture_classif()
  m1 <- LiblineaR(f$x, f$y, type = 0, bias = 1, epsilon = 1e-6)
  m5 <- LiblineaR(f$x, f$y, type = 0, bias = 5, epsilon = 1e-6)

  # Different bias magnitudes are different training data (a differently
  # scaled constant feature is appended), so the fitted bias coefficient
  # itself must differ -- if it didn't, the bias value would be being
  # ignored rather than incorporated into the design matrix.
  expect_false(isTRUE(all.equal(m1$W[, "Bias"], m5$W[, "Bias"])))
})
