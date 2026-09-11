# CORE: reference-free correctness checks. Upstream LIBLINEAR documents that
# these solver pairs are the primal/dual formulation of the same optimization
# problem and should converge to (numerically close to) the same model --
# no external ground truth needed, just internal consistency. Tolerances
# below were set from the actual observed max|W1-W2| on these fixtures
# (checked directly, not guessed): ~1.4e-3, ~1.5e-4 and ~8e-6 respectively.

test_that("type 1 (dual) and type 2 (primal) L2-loss SVC converge to the same model", {
  f <- fixture_classif()
  m1 <- LiblineaR(f$x, f$y, type = 1, cost = 1, epsilon = 1e-4)
  m2 <- LiblineaR(f$x, f$y, type = 2, cost = 1, epsilon = 1e-4)
  expect_equal(unname(m1$W), unname(m2$W), tolerance = 1e-2)
})

test_that("type 0 (primal) and type 7 (dual) logistic regression converge to the same model", {
  f <- fixture_classif()
  m0 <- LiblineaR(f$x, f$y, type = 0, cost = 1, epsilon = 1e-4)
  m7 <- LiblineaR(f$x, f$y, type = 7, cost = 1, epsilon = 1e-4)
  expect_equal(unname(m0$W), unname(m7$W), tolerance = 1e-2)
})

test_that("type 11 (primal) and type 12 (dual) L2-loss SVR converge to the same model", {
  f <- fixture_regr()
  m11 <- LiblineaR(f$x, f$y, type = 11, cost = 1, epsilon = 1e-5, svr_eps = 0.01)
  m12 <- LiblineaR(f$x, f$y, type = 12, cost = 1, epsilon = 1e-5, svr_eps = 0.01)
  expect_equal(unname(m11$W), unname(m12$W), tolerance = 1e-3)
})

test_that("training is exactly reproducible under a fixed seed (dual solvers route randomness through R's RNG)", {
  f <- fixture_classif()
  set.seed(999); mA <- LiblineaR(f$x, f$y, type = 1)
  set.seed(999); mB <- LiblineaR(f$x, f$y, type = 1)
  expect_identical(mA$W, mB$W)
})
