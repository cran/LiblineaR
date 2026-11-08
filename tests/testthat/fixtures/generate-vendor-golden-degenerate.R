# Fixture generator for tests/testthat/test-vendor-golden-degenerate.R
#
# NOT a test file: testthat never auto-runs anything under fixtures/, and
# this script is not sourced by any helper-*.R. Run it deliberately, by hand,
# from the package root:
#
#   Rscript tests/testthat/fixtures/generate-vendor-golden-degenerate.R
#
# WHAT THIS CAPTURES
# -------------------
# A golden-value regression battery for the vendored LIBLINEAR solver,
# pinned from the CURRENT, verified-correct solver (LiblineaR 2.10.26, all
# 888 pre-existing testthat assertions passing) BEFORE the planned v2.1 ->
# v2.50 vendor upgrade (src/linear.cpp, src/tron.cpp -> newton.cpp,
# trainLinear.c, predictLinear.c) begins.
#
# Unlike tests/testthat/fixtures/generate-vendor-golden-classification.R
# (breadth across cost/bias/epsilon on ordinary, well-behaved data), this
# fixture's job is DEPTH ON DEGENERACY: numerically stressful/edge-case
# design matrices and label distributions that sit near a convergence
# boundary or make the design matrix itself degenerate -- exactly where a
# solver rewrite (e.g. the v2.40 Newton-solver replacement of TRON) is most
# likely to produce a subtly different converged value even though
# "normal", well-conditioned cases pass fine.
#
# SCENARIOS
# ---------
#  1. near_separable       -- 2-class, huge margin, near-zero training error
#                              at the optimum. Stresses stopping-criterion
#                              behavior: does the solver detect convergence
#                              correctly when gradients vanish quickly,
#                              rather than iterating needlessly or stopping
#                              early/late?
#  2. overlap               -- 2-class, heavily overlapping/noisy, no clean
#                              margin. Stresses regularization-path
#                              robustness: no single dominant solution, so
#                              solver-internal tie-breaking/ordering changes
#                              are more likely to move the result.
#  3. highdim_clf/regr      -- p=50, n=15 (p >> n). Stresses L1-regularized
#                              types 5/6 (feature selection under extreme
#                              sparsity pressure) and the primal/dual choice
#                              differently, since dual solvers iterate over
#                              n dual variables while primal solvers iterate
#                              over p primal variables -- exactly reversed
#                              here vs. the ordinary n >> p case.
#  4. single_feat_clf/regr  -- p=1. The degenerate case where W has a single
#                              coefficient column and no cross-feature
#                              interaction can mask a per-feature indexing
#                              bug.
#  5. collinear             -- x3 is an EXACT linear combination of x1 and
#                              x2: a genuinely rank-deficient design matrix,
#                              the kind that can make a solver's internal
#                              Hessian/Gram-matrix handling numerically
#                              unstable even though the fitted decision
#                              boundary itself is well-defined.
#  6. boundary2/boundary3   -- the SAME underlying signal, once cut into 2
#                              classes and once into 3. Isolates a
#                              multiclass-dispatch-specific regression (e.g.
#                              in one-vs-rest bookkeeping, or type 4's own
#                              Crammer-Singer solver for boundary3) from a
#                              regression that would also show up in the
#                              2-class case.
#
# For each classification scenario, types {0,1,2,3,5,6,7} are pinned (all
# one-vs-rest-capable classification types); boundary3 additionally pins
# type 4 (Crammer-Singer multiclass, which needs 3+ classes). highdim and
# single_feat additionally pin all 3 regression types {11,12,13} against
# their own regression targets. That is 7 scenarios x 7 types (49) + 1
# extra type (boundary3's type 4) + 2 regression scenarios x 3 types (6) =
# 56 trained models, each contributing one pinned W coefficient matrix.
#
# RANDOMNESS
# ----------
# Dual solvers (types 1, 3, 4, 7, 12, 13) route randomness through R's own
# RNG (rand_between() in src/linear.cpp), so set.seed() is reset immediately
# before EVERY single LiblineaR() call below, not once at the top of the
# script.
#
# The datasets themselves are generated with their OWN fixed set.seed()
# calls (one per scenario, distinct from the per-call training seed and
# from each other) and are stored inside the .rds alongside the golden W
# matrices, so the test file re-trains against the exact same data rather
# than regenerating it from scratch -- it can never silently drift from
# what these golden values were pinned against.
#
# REGENERATING
# -------------
# Only re-run this script (overwriting vendor-golden-degenerate.rds) when a
# reviewed, understood change -- either to this generator itself, or to a
# vendor-upgrade stage that is meant to move these values -- calls for a
# new baseline. Never regenerate silently just to make a failing test pass;
# a failure here is the signal this battery exists to produce.

library(LiblineaR)

pkg_root <- normalizePath(".")
if (!file.exists(file.path(pkg_root, "DESCRIPTION"))) {
  stop("Run this script with the package root as the working directory.")
}

# --- Scenario 1: near-perfectly-separable 2-class data -------------------
# Large margin (means +-4, sd 0.4) -> near-zero training error at the
# optimum for every type.
make_near_separable <- function() {
  set.seed(9001)
  n <- 60
  x <- rbind(
    matrix(rnorm(n / 2 * 3, mean = -4, sd = 0.4), ncol = 3),
    matrix(rnorm(n / 2 * 3, mean =  4, sd = 0.4), ncol = 3)
  )
  colnames(x) <- c("x1", "x2", "x3")
  y <- factor(rep(c("neg", "pos"), each = n / 2))
  list(x = x, y = y)
}

# --- Scenario 2: heavily overlapping / noisy 2-class data -----------------
# Class means only 0.3 apart relative to sd=2 -> no clean margin exists.
make_overlap <- function() {
  set.seed(9002)
  n <- 60
  x <- rbind(
    matrix(rnorm(n / 2 * 3, mean = -0.15, sd = 2), ncol = 3),
    matrix(rnorm(n / 2 * 3, mean =  0.15, sd = 2), ncol = 3)
  )
  colnames(x) <- c("x1", "x2", "x3")
  y <- factor(rep(c("neg", "pos"), each = n / 2))
  list(x = x, y = y)
}

# --- Scenario 3: high-dimensional, low-sample-count data (p=50, n=15) -----
# Only the first 3 of 50 features carry signal; the rest is pure noise.
make_highdim <- function() {
  set.seed(9003)
  n <- 15; p <- 50
  x <- matrix(rnorm(n * p), nrow = n, ncol = p)
  colnames(x) <- paste0("x", seq_len(p))
  signal <- x[, 1] * 1.2 - x[, 2] * 0.8 + x[, 3] * 0.5
  y_clf <- factor(ifelse(signal > median(signal), "pos", "neg"))
  y_regr <- signal + rnorm(n, sd = 0.2)
  list(x = x, y_clf = y_clf, y_regr = y_regr)
}

# --- Scenario 4: single-feature data (p=1) ---------------------------------
make_single_feature <- function() {
  set.seed(9004)
  n <- 30
  x <- matrix(rnorm(n, sd = 2), ncol = 1)
  colnames(x) <- "x1"
  y_clf <- factor(ifelse(x[, 1] > median(x[, 1]), "pos", "neg"))
  y_regr <- as.vector(1.7 * x[, 1] - 0.4) + rnorm(n, sd = 0.15)
  list(x = x, y_clf = y_clf, y_regr = y_regr)
}

# --- Scenario 5: duplicated/collinear columns ------------------------------
# x3 is an exact linear combination of x1 and x2 -- rank-deficient design.
make_collinear <- function() {
  set.seed(9005)
  n <- 40
  x1 <- rnorm(n)
  x2 <- rnorm(n)
  x3 <- 2 * x1 - 0.5 * x2   # exact linear combination of x1, x2
  x4 <- rnorm(n)
  x <- cbind(x1, x2, x3, x4)
  y <- factor(ifelse(x1 + x4 > 0, "pos", "neg"))
  list(x = x, y = y)
}

# --- Scenario 6: same signal, cut into 2 classes vs 3 classes -------------
make_boundary <- function() {
  set.seed(9006)
  n <- 45
  x <- matrix(rnorm(n * 3), ncol = 3)
  colnames(x) <- c("x1", "x2", "x3")
  signal <- as.vector(x %*% c(1, -1, 0.5)) + rnorm(n, sd = 0.3)
  y2 <- factor(ifelse(signal > median(signal), "hi", "lo"))
  q <- quantile(signal, probs = c(1 / 3, 2 / 3))
  y3 <- factor(cut(signal, breaks = c(-Inf, q[1], q[2], Inf), labels = c("lo", "mid", "hi")))
  list(x = x, y2 = y2, y3 = y3)
}

ns <- make_near_separable()
ov <- make_overlap()
hd <- make_highdim()
sf <- make_single_feature()
cl <- make_collinear()
bd <- make_boundary()

CLF_TYPES  <- c(0, 1, 2, 3, 5, 6, 7)
CS_TYPES   <- c(0, 1, 2, 3, 4, 5, 6, 7)
REGR_TYPES <- c(11, 12, 13)

# TRAINING_SEED is reset immediately before every single LiblineaR() call
# (see RANDOMNESS above); it is not a "run the whole grid once" seed.
TRAINING_SEED <- 777

results <- list()

train_scenario <- function(x, y, types, scenario) {
  for (t in types) {
    key <- paste0(scenario, ".type", t)
    set.seed(TRAINING_SEED)
    m <- LiblineaR(x, y, type = t, cost = 1, svr_eps = 0.01)
    results[[key]] <<- unname(m$W)
  }
}

train_scenario(ns$x, ns$y, CLF_TYPES, "near_separable")
train_scenario(ov$x, ov$y, CLF_TYPES, "overlap")
train_scenario(hd$x, hd$y_clf, CLF_TYPES, "highdim_clf")
train_scenario(hd$x, hd$y_regr, REGR_TYPES, "highdim_regr")
train_scenario(sf$x, sf$y_clf, CLF_TYPES, "single_feat_clf")
train_scenario(sf$x, sf$y_regr, REGR_TYPES, "single_feat_regr")
train_scenario(cl$x, cl$y, CLF_TYPES, "collinear")
train_scenario(bd$x, bd$y2, CLF_TYPES, "boundary2")
train_scenario(bd$x, bd$y3, CS_TYPES, "boundary3")

golden <- list(
  meta = list(
    package_version = as.character(utils::packageVersion("LiblineaR")),
    generated = as.character(Sys.time()),
    clf_types = CLF_TYPES,
    cs_types = CS_TYPES,
    regr_types = REGR_TYPES,
    training_seed = TRAINING_SEED
  ),
  # Datasets are stored here (not regenerated inside the test file) so the
  # test can never silently drift from the exact data these golden values
  # were pinned against.
  datasets = list(
    near_separable = ns,
    overlap = ov,
    highdim = hd,
    single_feat = sf,
    collinear = cl,
    boundary = bd
  ),
  results = results
)

out_path <- file.path(pkg_root, "tests", "testthat", "fixtures", "vendor-golden-degenerate.rds")
saveRDS(golden, out_path, version = 2)

cat("Wrote", length(results), "golden scenarios to", out_path, "\n")
