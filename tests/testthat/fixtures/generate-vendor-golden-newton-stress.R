# Generator for tests/testthat/fixtures/vendor-golden-newton-stress.rds
#
# Plain R script -- NOT a test file, never auto-run by testthat. Run it
# deliberately, from the package root, with:
#   Rscript tests/testthat/fixtures/generate-vendor-golden-newton-stress.R
#
# WHAT THIS CAPTURES
# -------------------
# Vendor-upgrade Stage (ii) (v2.30 -> v2.40) replaced src/tron.cpp with
# src/newton.cpp: a real algorithm swap (trust-region Newton -> Armijo
# line-search Newton-CG) for the primal solvers of types 0 (L2R_LR), 2
# (L2R_L2LOSS_SVC) and 11 (L2R_L2LOSS_SVR) ONLY -- every dual/L1 type is
# confirmed bit-for-bit unaffected by this stage (verified directly against
# pristine v2.30 goldens, max|diff| == 0).
#
# A 7-agent risk analysis of this rewrite flagged one gap in the existing
# golden battery: no fixture tests BOTH of the following simultaneously,
# for types 0/2/11:
#   (a) near-separable / near-flat-gradient data, where the true gradient
#       at the optimum is tiny and a stopping-criterion rewrite is most
#       likely to under- or over-iterate before declaring convergence
#       (test-vendor-golden-degenerate.R's near_separable scenario covers
#       this axis alone, but always at the solver's default epsilon), AND
#   (b) a TIGHT epsilon (termination tolerance), which pushes the solver
#       to keep iterating deep into the flat-gradient region where TRON's
#       trust-region step-acceptance rule and Newton's Armijo line search
#       can plausibly diverge in how many extra iterations they take, and
#       therefore in the exact converged W
#       (test-vendor-golden-classweight-epsilon-bias.R covers tight
#       epsilon (1e-6) alone, but only on an ordinary 90/10-imbalanced
#       dataset that is not near-separable, and excludes type 11 entirely
#       since it is a classification-only fixture).
#
# This fixture crosses both axes at once, for exactly the 3 Newton-affected
# types, at TWO tight epsilon values (1e-6 and 1e-8, to bracket how tight
# is "tight enough" to expose a stopping-criterion difference) crossed with
# cost in {1, 100} (cost amplifies the primal objective's curvature/scale,
# changing how quickly the gradient norm decays near the optimum).
#
# DATASETS
# --------
# near_separable_stress: 2-class, 3-feature, means +-4 / sd 0.35 -- own
#   copy of the "near_separable" scenario pattern already used in
#   test-vendor-golden-degenerate.R (large margin -> near-zero training
#   error, vanishing gradient at the optimum), but with its own seed and
#   values, independent of that file's internal helper, per this stage's
#   task instructions.
# near_deterministic_regr: a regression target with a near-exact linear
#   relationship to 3 features (noise sd 0.001, three orders of magnitude
#   below the signal), so residuals -- and hence the L2-loss primal
#   objective's gradient -- are tiny almost everywhere at the optimum,
#   giving type 11 the same flat-gradient stress as the classification
#   scenario gives types 0/2.
#
# WHY THIS EXISTS / WHEN TO RE-RUN
# ---------------------------------
# This is a Stage (ii)-specific addition to the vendor-upgrade golden
# battery (v2.1 -> v2.50 program; see VENDOR-UPGRADE-PLAN.md). It pins
# values from the CURRENT, Stage-(ii)-merged solver (regularize_bias=1 and
# nu=0.5 fixes already applied, confirmed working) as the regression
# BASELINE for the remainder of the upgrade program, not externally-audited
# ground truth -- LIBLINEAR ships no independent reference harness to diff
# against (same caveat as every other vendor-golden-*.rds fixture in this
# project). Regenerate this .rds deliberately (never silently, never as a
# side effect of "tests failing") only once a further reviewed
# vendor-upgrade stage is meant to change these values.
#
# saveRDS(..., version = 2) is used explicitly below: the default
# serialize version 3 silently adds an undeclared R (>= 3.5.0) dependency,
# a real bug caught earlier in this project by R CMD build.

library(LiblineaR)

pkg_root <- normalizePath(".")
if (!file.exists(file.path(pkg_root, "DESCRIPTION"))) {
  stop("Run this script with the package root as the working directory.")
}

# --- near-separable classification data (own copy/seed) -------------------
make_near_separable_stress <- function() {
  set.seed(51050)
  n <- 60
  x <- rbind(
    matrix(rnorm(n / 2 * 3, mean = -4, sd = 0.35), ncol = 3),
    matrix(rnorm(n / 2 * 3, mean =  4, sd = 0.35), ncol = 3)
  )
  colnames(x) <- c("x1", "x2", "x3")
  y <- factor(rep(c("neg", "pos"), each = n / 2))
  list(x = x, y = y)
}

# --- near-deterministic regression target ----------------------------------
make_near_deterministic_regr <- function() {
  set.seed(51051)
  n <- 60
  x <- matrix(rnorm(n * 3), ncol = 3)
  colnames(x) <- c("x1", "x2", "x3")
  # Noise sd (0.001) three orders of magnitude below the signal's typical
  # scale, so residuals -- and the L2-loss gradient -- are tiny almost
  # everywhere at the optimum: the regression analogue of the
  # near-separable classification scenario's vanishing gradient.
  y <- as.vector(x %*% c(2, -1.5, 1)) + rnorm(n, sd = 0.001)
  list(x = x, y = y)
}

ns <- make_near_separable_stress()
nd <- make_near_deterministic_regr()

EPSILONS <- c(1e-6, 1e-8)
COSTS <- c(1, 100)
SVR_EPS <- 0.001  # tight insensitive-tube width, consistent with the
                   # near-deterministic target's near-zero residuals.

# All three types trained here (0, 2, 11) are Newton/TRON PRIMAL solvers,
# which do not route through R's RNG (rand_between() in src/linear.cpp) the
# way dual solvers do. set.seed() is nonetheless reset immediately before
# EVERY LiblineaR() call below, matching the convention used uniformly
# across this golden battery's other generator scripts (harmless here, and
# keeps this script's pattern identical to the others' regardless of type).
TRAINING_SEED <- 777

results <- list()

for (eps in EPSILONS) {
  for (cost in COSTS) {
    key <- paste0("type0_eps", format(eps, scientific = TRUE), "_cost", cost)
    set.seed(TRAINING_SEED)
    m <- LiblineaR(ns$x, ns$y, type = 0, cost = cost, epsilon = eps)
    results[[key]] <- unname(m$W)

    key <- paste0("type2_eps", format(eps, scientific = TRUE), "_cost", cost)
    set.seed(TRAINING_SEED)
    m <- LiblineaR(ns$x, ns$y, type = 2, cost = cost, epsilon = eps)
    results[[key]] <- unname(m$W)

    key <- paste0("type11_eps", format(eps, scientific = TRUE), "_cost", cost)
    set.seed(TRAINING_SEED)
    m <- LiblineaR(nd$x, nd$y, type = 11, cost = cost, epsilon = eps,
                    svr_eps = SVR_EPS)
    results[[key]] <- unname(m$W)
  }
}

golden <- list(
  meta = list(
    package_version = as.character(utils::packageVersion("LiblineaR")),
    generated = as.character(Sys.time()),
    epsilons = EPSILONS,
    costs = COSTS,
    svr_eps = SVR_EPS,
    training_seed = TRAINING_SEED
  ),
  # Datasets stored alongside the golden W matrices so the test file
  # re-trains against the exact same data rather than regenerating it from
  # scratch -- it can never silently drift from what these golden values
  # were pinned against.
  datasets = list(
    near_separable_stress = ns,
    near_deterministic_regr = nd
  ),
  results = results
)

out_path <- file.path(pkg_root, "tests", "testthat", "fixtures", "vendor-golden-newton-stress.rds")
saveRDS(golden, out_path, version = 2)

cat("Wrote", length(results), "golden values to", out_path, "\n")
