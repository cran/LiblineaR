# Fixture generator for tests/testthat/test-vendor-golden-classweight-epsilon-bias.R
#
# NOT a test file: testthat never auto-runs files under fixtures/, and this
# script is not sourced by any test. It is a plain R script, run manually
# and deliberately (never automatically) to (re)produce
# tests/testthat/fixtures/vendor-golden-classweight-epsilon-bias.rds.
#
# WHAT THIS CAPTURES
# -------------------
# A golden-value regression grid for the parameters most directly tied to
# solver convergence behaviour -- class weights (wi), the termination
# tolerance (epsilon), and the bias term -- crossed over the primal/dual
# L2-regularized logistic/SVM solver family (type in 0:3). These are exactly
# the parameters implicated in the 2.10-25 epsilon-default bug and its
# parsnip reverse-dependency break, so this grid is deliberately broad
# rather than deep on any one combination.
#
# It pins fitted W coefficients from the CURRENT, already-verified-correct
# vendored LIBLINEAR solver (~v2.1) across:
#   type    in {0, 1, 2, 3}                              (L2R_LR, L2R_L2LOSS_SVC_DUAL, L2R_L2LOSS_SVC, L2R_L1LOSS_SVC_DUAL)
#   wi      in {NULL, 5x minority weight, 20x minority weight}
#   epsilon in {NULL, 0.1, 0.01, 1e-6}
#   bias    in {-1, 1}
# on a fixed, deterministic 90/10 imbalanced 2-class dataset (own copy, not
# shared with test-class-weights.R, for independence/continuity of this
# battery specifically).
#
# It also separately pins a same-vs-explicit-default-epsilon pair (epsilon
# = NULL vs epsilon = the current documented per-solver default, per
# @param epsilon in R/LiblineaR.R: 0.01 for primal types 0/2, 0.1 for dual
# types 1/3) to freeze the invariant that NULL and "the actual default"
# produce IDENTICAL W -- exactly the invariant a solver-default change (like
# the 2.10-25 bug) could silently break.
#
# WHY THIS EXISTS / WHEN TO RE-RUN
# ---------------------------------
# This is the safety net for the upcoming multi-stage vendor upgrade of the
# bundled LIBLINEAR C/C++ source (src/linear.cpp, src/tron.cpp -> newton.cpp,
# trainLinear.c, predictLinear.c) from ~v2.1 to v2.50. After each vendor-
# upgrade stage, re-running test-vendor-golden-classweight-epsilon-bias.R
# against this frozen fixture should catch ANY numeric drift in these
# solvers' outputs -- whether an expected, reviewed change (e.g. the v2.40
# Newton-solver rewrite legitimately moving converged values within
# tolerance) or a genuine regression.
#
# Like tests/testthat/fixtures/reference-coefficients.rds, this fixture
# freezes coefficients from a verified build as the regression BASELINE,
# not externally-audited ground truth (LIBLINEAR ships no independent
# reference harness to diff against). Regenerate this .rds deliberately
# (never silently, never as a side effect of "tests failing") by re-running
# this exact script, only once a reviewed vendor-upgrade stage is meant to
# change these values, and only after confirming the new values by other
# means (e.g. comparison against upstream LIBLINEAR's own reference output
# for the same inputs).
#
# HOW TO RUN
# ----------
# From the package root (working directory = package root):
#   Rscript tests/testthat/fixtures/generate-vendor-golden-classweight-epsilon-bias.R
#
# This requires the package to already be installed (library(LiblineaR)),
# matching how the rest of this test suite is run.

library(LiblineaR)

# --- Fixed, deterministic 90/10 imbalanced 2-class dataset -----------------
# Own copy of the 90/10-style construction used in test-class-weights.R, for
# continuity of *pattern* only -- this script does not source that file, and
# its seed and exact values are independent so this fixture's provenance is
# self-contained.
make_fixture <- function() {
  set.seed(2026)
  n_maj <- 90; n_min <- 10
  x <- rbind(
    matrix(rnorm(n_maj * 2, mean =  0.4), ncol = 2),
    matrix(rnorm(n_min * 2, mean = -0.4), ncol = 2)
  )
  colnames(x) <- c("x1", "x2")
  y <- factor(c(rep("majority", n_maj), rep("minority", n_min)))
  list(x = x, y = y)
}

f <- make_fixture()

types <- c(0, 1, 2, 3)
wi_list <- list(
  unweighted = NULL,
  w5  = c(minority = 5),
  w20 = c(minority = 20)
)
epsilon_list <- list(
  default = NULL,
  e0.1    = 0.1,
  e0.01   = 0.01,
  e1em6   = 1e-6
)
bias_list <- c(-1, 1)

# Per-solver documented default epsilon (R/LiblineaR.R @param epsilon):
# primal types (0, 2, 5, 6) default to 0.01; dual types (1, 3, 4, 7, 12, 13)
# default to 0.1. Only types 0:3 are exercised by this dimension.
default_epsilon_for_type <- function(type) {
  if (type %in% c(0, 2)) 0.01 else 0.1
}

results <- list()

# Dual solvers (types 1, 3, 4, 7, 12, 13) route randomness through R's own
# RNG (rand_between() in src/linear.cpp), so set.seed() is called
# immediately before EVERY LiblineaR() call below, not once at the top of
# this script -- otherwise runs would not be reproducible across the
# generator and the test file re-running the same grid.

for (type in types) {
  for (wi_name in names(wi_list)) {
    for (eps_name in names(epsilon_list)) {
      for (bias in bias_list) {
        key <- paste("type", type, "wi", wi_name, "eps", eps_name, "bias", bias, sep = "_")
        set.seed(4242)
        m <- LiblineaR(
          f$x, f$y,
          type = type,
          cost = 1,
          epsilon = epsilon_list[[eps_name]],
          bias = bias,
          wi = wi_list[[wi_name]]
        )
        results[[key]] <- unname(m$W)
      }
    }
  }
}

# --- Explicit NULL-vs-documented-default-epsilon invariant ------------------
# One pinned pair per type, at default wi/bias, freezing that
# epsilon = NULL and epsilon = <documented default for that solver family>
# produce IDENTICAL W. This is the exact invariant a solver-default change
# (like the 2.10-25 epsilon-default bug) could silently break.
for (type in types) {
  key_null <- paste0("defaultcheck_type", type, "_null")
  key_explicit <- paste0("defaultcheck_type", type, "_explicit")

  set.seed(4242)
  m_null <- LiblineaR(f$x, f$y, type = type, cost = 1, epsilon = NULL, bias = 1)
  results[[key_null]] <- unname(m_null$W)

  set.seed(4242)
  m_explicit <- LiblineaR(f$x, f$y, type = type, cost = 1,
                           epsilon = default_epsilon_for_type(type), bias = 1)
  results[[key_explicit]] <- unname(m_explicit$W)
}

out_path <- file.path("tests", "testthat", "fixtures", "vendor-golden-classweight-epsilon-bias.rds")
saveRDS(results, out_path, version = 2)
cat("Wrote", length(results), "entries to", out_path, "\n")
