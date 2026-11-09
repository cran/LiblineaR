# VENDOR-UPGRADE SAFETY NET: golden values for predict.LiblineaR() output
# (probabilities and decisionValues), pinned from the current, verified
# solver ahead of the LIBLINEAR vendor upgrade (~v2.1 -> v2.50). This
# exercises predictLinear.c/do_predict() across a real type x cost x bias
# grid on both a 2-class and a 3-class fixture -- a code path the upgrade
# could touch independently of the training side (trainLinear.c/
# linear.cpp), and which test-predict-proba-decision.R only checks
# structurally (sums to 1, sign-consistency), not against exact pinned
# values. See fixtures/generate-vendor-golden-predict.R for full
# provenance and how to deliberately regenerate the .rds after a reviewed
# change is meant to alter these values.
#
# Own, self-contained fixtures (duplicated from helper-fixtures.R's
# fixture_classif() for the 2-class case, and a new 3-class fixture),
# matching the generator script exactly, so this dimension's coverage
# stays independent of other dimensions' files. Fresh models are trained
# here and compared against the frozen fixture -- a manual rerun during
# generation showed exact (0) numeric difference in this environment, so
# tolerance=1e-6 (this project's usual CORE-file tolerance, see
# test-reference-coefficients.R) is used to allow for possible cross-
# platform/BLAS floating-point variation without being loose enough to
# hide a real regression.

fixture_classif2 <- function() {
  set.seed(123)
  n <- 40
  x <- rbind(
    matrix(rnorm(n / 2 * 3, mean = -1.5), ncol = 3),
    matrix(rnorm(n / 2 * 3, mean =  1.5), ncol = 3)
  )
  colnames(x) <- c("x1", "x2", "x3")
  y <- factor(rep(c("neg", "pos"), each = n / 2))
  list(x = x, y = y)
}

fixture_classif3 <- function() {
  set.seed(456)
  n <- 60
  x <- rbind(
    matrix(rnorm(n / 3 * 3, mean = -4), ncol = 3),
    matrix(rnorm(n / 3 * 3, mean =  0), ncol = 3),
    matrix(rnorm(n / 3 * 3, mean =  4), ncol = 3)
  )
  colnames(x) <- c("x1", "x2", "x3")
  y <- factor(rep(c("a", "b", "c"), each = n / 3))
  list(x = x, y = y)
}

# Must match generate-vendor-golden-predict.R exactly: same values, same
# order, so the index-based keys line up with the frozen fixture.
GOLDEN_COSTS  <- c(0.01, 1, 100)
GOLDEN_BIASES <- c(-1, 1)
# Exact same-platform pin. Portability is handled by expect_golden() in
# helper-golden-portability.R: scenarios whose solver converged are pinned
# at this tolerance everywhere, while a scenario that stops at its iteration
# cap is asserted on invariants only on CRAN and pinned exactly off it.
# Several type-3 scenarios here are capped, and at least two of them
# (cost=2 and cost=3 at bias=2) are bistable: CRAN's arm64 legs and a local
# FMA build return a different basin entirely. No per-scenario registry is
# needed any more -- the capped state is detected at run time.
GOLDEN_TOL <- 1e-6

golden_key <- function(fname, type, ci, bi) paste(fname, "type", type, "cost", ci, "bias", bi, sep = "_")

test_that("proba=TRUE probability matrices match frozen golden values across type x cost x bias (2-class)", {
  # Risk covered: any change to the logistic-regression probability
  # computation in predictLinear.c/do_predict() (e.g. a sigmoid/normalization
  # rewrite that ships with the vendor upgrade) that shifts computed
  # probabilities without breaking the "sums to 1" / sign-consistency
  # structural checks in test-predict-proba-decision.R.
  golden <- readRDS(test_path("fixtures", "vendor-golden-predict.rds"))
  f <- fixture_classif2()

  for (type in c(0, 6, 7)) {
    for (ci in seq_along(GOLDEN_COSTS)) {
      for (bi in seq_along(GOLDEN_BIASES)) {
        set.seed(42)
        g <- ll_golden_fit(data = f$x, target = f$y, type = type,
                           cost = GOLDEN_COSTS[ci], bias = GOLDEN_BIASES[bi])
        p <- predict(g$model, f$x, proba = TRUE)
        k <- golden_key("classif2", type, ci, bi)
        expect_golden(p$probabilities, golden$probabilities[[k]],
                      key = k, fit = g, exact_tol = GOLDEN_TOL)
      }
    }
  }
})

test_that("proba=TRUE probability matrices match frozen golden values across type x cost x bias (3-class)", {
  # Risk covered: same as above, but on a 3-class fixture -- the
  # multiclass logistic normalization branch (nr_class>2) in
  # predictLinear.c/do_predict() is a separate code path from the 2-class
  # one and could drift independently.
  golden <- readRDS(test_path("fixtures", "vendor-golden-predict.rds"))
  f <- fixture_classif3()

  for (type in c(0, 6, 7)) {
    for (ci in seq_along(GOLDEN_COSTS)) {
      for (bi in seq_along(GOLDEN_BIASES)) {
        set.seed(42)
        g <- ll_golden_fit(data = f$x, target = f$y, type = type,
                           cost = GOLDEN_COSTS[ci], bias = GOLDEN_BIASES[bi])
        p <- predict(g$model, f$x, proba = TRUE)
        k <- golden_key("classif3", type, ci, bi)
        expect_golden(p$probabilities, golden$probabilities[[k]],
                      key = k, fit = g, exact_tol = GOLDEN_TOL)
      }
    }
  }
})

test_that("decisionValues=TRUE matrices match frozen golden values across type x cost x bias (2-class)", {
  # Risk covered: any change to the raw discriminant-value computation in
  # predictLinear.c/do_predict() (w.x + bias*b per class) across ALL
  # classification types, including type 4 (Crammer-Singer) which uses the
  # multiclass code path (2 nonzero columns) even with only 2 classes --
  # unlike the other 2-class types, which only ever populate one column
  # (see test-predict-proba-decision.R).
  golden <- readRDS(test_path("fixtures", "vendor-golden-predict.rds"))
  f <- fixture_classif2()

  for (type in 0:7) {
    for (ci in seq_along(GOLDEN_COSTS)) {
      for (bi in seq_along(GOLDEN_BIASES)) {
        set.seed(42)
        g <- ll_golden_fit(data = f$x, target = f$y, type = type,
                           cost = GOLDEN_COSTS[ci], bias = GOLDEN_BIASES[bi])
        pd <- predict(g$model, f$x, decisionValues = TRUE)
        k <- golden_key("classif2", type, ci, bi)
        expect_golden(pd$decisionValues, golden$decisionValues[[k]],
                      key = k, fit = g, exact_tol = GOLDEN_TOL)
      }
    }
  }
})

test_that("decisionValues=TRUE matrices match frozen golden values across type x cost x bias (3-class)", {
  # Risk covered: same as above, on a genuinely multiclass (3-class)
  # fixture, where every type (not just type 4) takes the nr_class>2
  # branch of predict_values() / do_predict() and produces a full k-column
  # decision-value matrix -- a shape the 2-class fixture above cannot
  # exercise for types 0,1,2,3,5,6,7.
  golden <- readRDS(test_path("fixtures", "vendor-golden-predict.rds"))
  f <- fixture_classif3()

  for (type in 0:7) {
    for (ci in seq_along(GOLDEN_COSTS)) {
      for (bi in seq_along(GOLDEN_BIASES)) {
        set.seed(42)
        g <- ll_golden_fit(data = f$x, target = f$y, type = type,
                           cost = GOLDEN_COSTS[ci], bias = GOLDEN_BIASES[bi])
        pd <- predict(g$model, f$x, decisionValues = TRUE)
        k <- golden_key("classif3", type, ci, bi)
        expect_golden(pd$decisionValues, golden$decisionValues[[k]],
                      key = k, fit = g, exact_tol = GOLDEN_TOL)
      }
    }
  }
})
