# Generator for tests/testthat/fixtures/vendor-golden-predict.rds.
#
# WHAT THIS CAPTURES: golden (frozen) predict.LiblineaR() OUTPUT VALUES --
# not training-side coefficients (see the classification/regression-grid
# dimensions' own generators for those) -- across a real parameter grid, so
# that predictLinear.c/do_predict() has its own independent regression
# safety net for the upcoming LIBLINEAR vendor upgrade (currently ~v2.1,
# going to v2.50), which could touch the prediction code path (decision
# value / probability computation) independently of the training path
# (trainLinear.c/linear.cpp). Two output surfaces are pinned:
#   - predict(..., proba=TRUE)$probabilities   for type in {0, 6, 7}
#     (the only types LIBLINEAR supports probability estimates for)
#   - predict(..., decisionValues=TRUE)$decisionValues  for type in 0:7
#     (all classification types)
# crossed with cost in {0.01, 1, 100} (regularization strength) and
# bias in {-1, 1} (bias term off/on), on both a 2-class and a 3-class
# fixture, so the grid exercises binary vs. multiclass code paths
# (including type 4, Crammer-Singer, which is multiclass-shaped even with
# only 2 classes -- see LiblineaR.R's isMulticlass logic).
#
# This is a REGRESSION baseline frozen from a verified-correct build (the
# 888 pre-existing testthat assertions all pass on this solver), not
# externally-audited ground truth -- LIBLINEAR ships no independent
# reference harness to diff against (same situation as
# reference-coefficients.rds).
#
# DO NOT re-run this automatically. Re-run it deliberately, with
# Rscript (working directory = package root):
#   Rscript tests/testthat/fixtures/generate-vendor-golden-predict.R
# only when a REVIEWED change (e.g. a vendor-upgrade stage) is meant to
# alter these values, and diff the resulting .rds against git history to
# confirm the change is the expected one before committing it.
#
# This script is self-contained: it does not source helper-fixtures.R (so
# this dimension's fixtures stay independent of other dimensions' files),
# and it is never picked up by testthat (files here are only loaded when a
# test file explicitly test_path()s the .rds, and testthat only
# auto-sources tests/testthat/helper-*.R and test-*.R -- never
# fixtures/*.R).

library(LiblineaR)

# A small, fixed, cleanly separable 2-class dataset -- deliberately easy to
# separate so different types converge to comparable decision boundaries.
# Copy of helper-fixtures.R's fixture_classif(), duplicated here (rather
# than sourced) so this dimension's fixture generation stays fully
# self-contained and independent of other dimensions' files.
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

# A small, fixed, cleanly separable 3-class dataset, own copy independent of
# any other dimension's 3-class fixture.
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

# Parameter grid. Indices (ci, bi), not the numeric values themselves, are
# used to build result keys below, so key construction can never drift due
# to floating point formatting differences between this script and the
# test file that re-reads it.
costs  <- c(0.01, 1, 100)
biases <- c(-1, 1)
types_all   <- 0:7          # all classification types: decisionValues grid
types_proba <- c(0, 6, 7)   # logistic regression types only: proba grid

fixtures <- list(classif2 = fixture_classif2(), classif3 = fixture_classif3())

key <- function(fname, type, ci, bi) paste(fname, "type", type, "cost", ci, "bias", bi, sep = "_")

golden <- list(decisionValues = list(), probabilities = list(), costs = costs, biases = biases)

for (fname in names(fixtures)) {
  f <- fixtures[[fname]]
  for (type in types_all) {
    for (ci in seq_along(costs)) {
      for (bi in seq_along(biases)) {
        # Dual solvers (types 1,3,4,7,12,13) route randomness through R's
        # own RNG (rand_between in src/linear.cpp): seed immediately before
        # THIS training call, not once outside the loop, or the run would
        # not be reproducible run-to-run or file-to-file.
        set.seed(42)
        m <- LiblineaR(f$x, f$y, type = type, cost = costs[ci], bias = biases[bi])

        k <- key(fname, type, ci, bi)

        pd <- predict(m, f$x, decisionValues = TRUE)
        golden$decisionValues[[k]] <- pd$decisionValues

        if (type %in% types_proba) {
          pp <- predict(m, f$x, proba = TRUE)
          golden$probabilities[[k]] <- pp$probabilities
        }
      }
    }
  }
}

saveRDS(golden, file.path("tests", "testthat", "fixtures", "vendor-golden-predict.rds"), version = 2)
cat("Wrote", length(golden$decisionValues), "decisionValues matrices and",
    length(golden$probabilities), "probability matrices to vendor-golden-predict.rds\n")
