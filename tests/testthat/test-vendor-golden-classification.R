# Golden-value regression battery for the vendored LIBLINEAR solver, ahead
# of the planned v2.1 -> v2.50 vendor-upgrade program (src/linear.cpp,
# src/tron.cpp -> newton.cpp, trainLinear.c, predictLinear.c). NOT a
# reference-correctness check like test-reference-coefficients.R (which it
# does not duplicate -- that file only ever uses cost=1, no bias/epsilon
# variation, no multiclass): this file pins a much wider classification
# parameter grid so any vendor-upgrade stage that moves numeric output --
# expected (e.g. the documented v2.40 Newton-solver rewrite legitimately
# changing converged values within a re-justified tolerance) or a genuine
# regression -- gets caught and has to be consciously reviewed.
#
# Grid: all 8 classification types (0-7) x cost {0.01, 1, 100} x
# bias {-1 (off), 1, 5} x epsilon {NULL default, 1e-6 tight} x dataset
# {fixture_classif() 2-class, a dedicated 3-class Gaussian-blob dataset}.
# See tests/testthat/fixtures/generate-vendor-golden-classification.R for
# the full grid rationale and how to (deliberately) regenerate the fixture.
#
# Reproducibility: re-running this exact grid against the current solver
# reproduces every one of the 288 frozen W matrices bit-for-bit (observed
# max|W_now - W_golden| == 0 across the whole grid, checked directly before
# picking a tolerance). tolerance=1e-6 below is therefore generous headroom
# for build/BLAS-level float noise, not a threshold tuned to just barely
# pass -- any real drift will be many orders of magnitude above it.
#
# Dual solvers (types 1, 3, 4, 7) route randomness through R's own RNG
# (rand_between() in src/linear.cpp), so set.seed() is reset immediately
# before every single LiblineaR() call in the loop below, matching how the
# fixture itself was generated.

GOLDEN_TOL <- 1e-6
TRAINING_SEED <- 42

golden <- readRDS(test_path("fixtures", "vendor-golden-classification.rds"))

fc <- fixture_classif()
datasets <- list(binary = fc, multiclass3 = golden$multiclass3)

costs     <- c(0.01, 1, 100)
biases    <- c(-1, 1, 5)
eps_modes <- list(default = NULL, tight = 1e-6)

# Runs one (cost x bias x dataset x epsilon-mode) sub-grid for a single
# solver type against the frozen golden values, and returns the number of
# scenarios checked (used only to sanity-check grid coverage below).
run_type_grid <- function(type) {
  n <- 0
  for (cost in costs) {
    for (bias in biases) {
      for (ds_name in names(datasets)) {
        ds <- datasets[[ds_name]]
        for (eps_name in names(eps_modes)) {
          eps <- eps_modes[[eps_name]]
          key <- paste("type", type, "cost", cost, "bias", bias, ds_name, eps_name, sep = "_")

          set.seed(TRAINING_SEED)
          m <- LiblineaR(ds$x, ds$y, type = type, cost = cost, bias = bias, epsilon = eps)

          expect_equal(unname(m$W), golden$results[[key]], tolerance = GOLDEN_TOL, label = key)
          n <- n + 1
        }
      }
    }
  }
  n
}

test_that("type 0 (L2-regularized logistic regression, primal) golden grid", {
  # Risk: this is the primal Newton/trust-region solver path
  # (src/tron.cpp -> newton.cpp under the vendor upgrade) -- the exact
  # optimizer the planned v2.40 rewrite targets first.
  n <- run_type_grid(0)
  expect_equal(n, 36)
})

test_that("type 1 (L2-regularized L2-loss SVC, dual) golden grid", {
  # Risk: dual coordinate-descent solver; also exercises the
  # rand_between()/R-RNG interaction the vendor upgrade could disturb
  # (e.g. by changing the order or count of random draws per iteration).
  n <- run_type_grid(1)
  expect_equal(n, 36)
})

test_that("type 2 (L2-regularized L2-loss SVC, primal) golden grid", {
  # Risk: shares the primal Newton/trust-region code with type 0, but on a
  # different (non-differentiable-at-the-margin) L2-loss objective -- a
  # regression that only affects one loss family would show here and not
  # in type 0.
  n <- run_type_grid(2)
  expect_equal(n, 36)
})

test_that("type 3 (L2-regularized L1-loss SVC, dual) golden grid", {
  # Risk: dual coordinate descent on the non-differentiable hinge (L1) loss
  # -- a distinct subgradient-handling path from type 1's L2-loss dual.
  n <- run_type_grid(3)
  expect_equal(n, 36)
})

test_that("type 4 (Crammer-Singer multiclass SVM) golden grid", {
  # Risk: its own dedicated multiclass dual solver (not one-vs-rest), so a
  # regression here is invisible to every other type's grid; the
  # multiclass3 dataset rows are what actually stress this path.
  n <- run_type_grid(4)
  expect_equal(n, 36)
})

test_that("type 5 (L1-regularized L2-loss SVC) golden grid", {
  # Risk: L1-regularized coordinate-descent (CDN) solver -- a code path
  # entirely separate from the L2-regularized primal/dual solvers above,
  # including its own sparsity-inducing convergence criterion.
  n <- run_type_grid(5)
  expect_equal(n, 36)
})

test_that("type 6 (L1-regularized logistic regression) golden grid", {
  # Risk: L1-regularized coordinate descent applied to the logistic loss --
  # shares the CDN machinery with type 5 but a different loss/gradient, so
  # a regression could affect one and not the other.
  n <- run_type_grid(6)
  expect_equal(n, 36)
})

test_that("type 7 (L2-regularized logistic regression, dual) golden grid", {
  # Risk: dual coordinate descent for the logistic loss -- the dual
  # counterpart to type 0's primal solver; a Newton-rewrite regression
  # confined to the primal side would leave this one unaffected, which is
  # itself useful signal when triaging a failure.
  n <- run_type_grid(7)
  expect_equal(n, 36)
})
