# Fixture generator for tests/testthat/test-vendor-golden-classification.R
#
# NOT a test file: testthat never auto-runs anything under fixtures/, and
# this script is not sourced by any helper-*.R. Run it deliberately, by hand,
# from the package root:
#
#   Rscript tests/testthat/fixtures/generate-vendor-golden-classification.R
#
# WHAT THIS CAPTURES
# -------------------
# A golden-value regression battery for the vendored LIBLINEAR C/C++ solver
# (src/linear.cpp etc.), pinned from the CURRENT, verified-correct solver
# (LiblineaR 2.10.26, all 888 pre-existing testthat assertions passing)
# BEFORE the planned v2.1 -> v2.50 vendor upgrade of that code begins. It
# exists purely as a safety net for that upgrade: after each upgrade stage,
# re-running test-vendor-golden-classification.R against this frozen fixture
# should show either (a) no numeric drift, or (b) a drift that a reviewer
# consciously expects (e.g. the documented v2.40 Newton-solver rewrite
# legitimately moving converged values within a wider, re-justified
# tolerance) -- never a silent, unexamined change.
#
# It intentionally does NOT duplicate the single-point coverage already in
# test-reference-coefficients.R (cost=1 only), test-solver-equivalence.R
# (primal/dual pairs, no golden pinning) or test-sparse-dense-equivalence.R
# (type 0 only, sparse vs dense). This fixture's job is breadth across a
# realistic classification parameter grid: all 8 classification solver
# types (0-7) x cost x bias x epsilon-mode x dataset.
#
# GRID
# ----
# types:   0:7 (all classification types -- primal/dual, L1/L2 reg,
#          L1/L2-loss SVC, logistic regression, and Crammer-Singer
#          multiclass SVM)
# cost:    0.01, 1, 100 (spans the regularization/fit trade-off)
# bias:    -1 (no bias/intercept term), 1, 5 (default-ish, and a larger
#          bias-feature scale that stresses the bias column differently)
# epsilon: NULL (documented LIBLINEAR per-type default) and 1e-6 (a single
#          tight, explicit epsilon) -- pinning both so a vendor-upgrade
#          change to what "default" resolves to for a given type is itself
#          caught, not just changes at a fixed epsilon.
# dataset: the shared 2-class fixture_classif() (see helper-fixtures.R,
#          n=40, 3 features) exercises the ordinary binary one-vs-rest path;
#          a new, self-contained 3-class dataset (below) exercises genuine
#          multiclass one-vs-rest dispatch (types 0,1,2,3,5,6,7) AND the
#          type-4 Crammer-Singer multiclass path, neither of which the
#          2-class fixture can distinguish from the binary case.
#
# That is 8 x 3 x 3 x 2 x 2 = 288 trained models, each contributing one
# pinned W coefficient matrix.
#
# RANDOMNESS
# ----------
# Dual solvers (types 1, 3, 4, 7) route randomness through R's own RNG
# (rand_between() in src/linear.cpp), so set.seed() is called immediately
# before EVERY single LiblineaR() call below, not once at the top of the
# script -- otherwise consecutive calls in the same grid would silently
# consume different amounts of RNG state and stop being reproducible.
#
# REGENERATING
# -------------
# Only re-run this script (overwriting vendor-golden-classification.rds)
# when a reviewed, understood change -- either to this generator itself, or
# to a vendor-upgrade stage that is meant to move these values -- calls for
# a new baseline. Never regenerate silently just to make a failing test
# pass; a failure here is the signal this battery exists to produce.

library(LiblineaR)

pkg_root <- normalizePath(".")
if (!file.exists(file.path(pkg_root, "DESCRIPTION"))) {
  stop("Run this script with the package root as the working directory.")
}

source(file.path(pkg_root, "tests", "testthat", "helper-fixtures.R"))

# --- 3-class fixture: 3 well-separated Gaussian blobs -----------------
# n=60 (20/class), 3 features, own fixed seed (distinct from the seeds used
# by fixture_classif()/fixture_regr() and from the per-call training seed
# below), so it is deterministic and reproducible independently of them.
make_fixture_multiclass3 <- function() {
  set.seed(555)
  n_per_class <- 20
  x <- rbind(
    matrix(rnorm(n_per_class * 3, mean = c(-4, -4, -4)), ncol = 3, byrow = TRUE),
    matrix(rnorm(n_per_class * 3, mean = c( 4,  0, -4)), ncol = 3, byrow = TRUE),
    matrix(rnorm(n_per_class * 3, mean = c( 0,  4,  4)), ncol = 3, byrow = TRUE)
  )
  colnames(x) <- c("x1", "x2", "x3")
  y <- factor(rep(c("A", "B", "C"), each = n_per_class))
  list(x = x, y = y)
}

fc <- fixture_classif()
fm <- make_fixture_multiclass3()

datasets <- list(binary = fc, multiclass3 = fm)

types    <- 0:7
costs    <- c(0.01, 1, 100)
biases   <- c(-1, 1, 5)
eps_modes <- list(default = NULL, tight = 1e-6)

# TRAINING_SEED is reset immediately before every single LiblineaR() call
# (see RANDOMNESS above); it is not a "run the whole grid once" seed.
TRAINING_SEED <- 42

results <- list()

for (t in types) {
  for (cost in costs) {
    for (bias in biases) {
      for (ds_name in names(datasets)) {
        ds <- datasets[[ds_name]]
        for (eps_name in names(eps_modes)) {
          eps <- eps_modes[[eps_name]]
          key <- paste("type", t, "cost", cost, "bias", bias, ds_name, eps_name, sep = "_")

          set.seed(TRAINING_SEED)
          m <- LiblineaR(ds$x, ds$y, type = t, cost = cost, bias = bias, epsilon = eps)

          results[[key]] <- unname(m$W)
        }
      }
    }
  }
}

golden <- list(
  meta = list(
    package_version = as.character(utils::packageVersion("LiblineaR")),
    generated = as.character(Sys.time()),
    types = types,
    costs = costs,
    biases = biases,
    eps_modes = names(eps_modes),
    training_seed = TRAINING_SEED
  ),
  # The multiclass3 dataset itself is stored here, not regenerated inside
  # the test file, so the test can never silently drift from the exact
  # data these golden values were pinned against.
  multiclass3 = fm,
  results = results
)

out_path <- file.path(pkg_root, "tests", "testthat", "fixtures", "vendor-golden-classification.rds")
saveRDS(golden, out_path, version = 2)

cat("Wrote", length(results), "golden scenarios to", out_path, "\n")
