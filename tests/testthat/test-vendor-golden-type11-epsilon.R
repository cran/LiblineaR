# VENDOR-UPGRADE GOLDEN BATTERY: type 11 (L2R_L2LOSS_SVR primal) epsilon
# (termination-tolerance) sensitivity, crossed with cost, on two datasets.
#
# WHY THIS FILE EXISTS
# ---------------------
# Vendor-upgrade Stage (ii) (v2.30 -> v2.40) replaced upstream's TRON
# trust-region solver with a Newton-CG + Armijo line-search solver
# (src/tron.cpp -> src/newton.cpp) for the three PRIMAL solver types 0
# (L2R_LR), 2 (L2R_L2LOSS_SVC) and 11 (L2R_L2LOSS_SVR) -- a real algorithm
# replacement, not a refactor. The termination tolerance (epsilon /
# param.eps) is exactly what the new solver's stopping test consumes, so
# it is the parameter most likely to expose a behavioural difference
# between the old and new code paths -- yet before this file, epsilon
# sensitivity for type 11 had ZERO golden coverage anywhere in this
# battery:
#   - test-vendor-golden-classweight-epsilon-bias.R varies epsilon but only
#     for types 0, 1, 2, 3 (classification); type 11 is absent.
#   - test-vendor-golden-svr-paramgrid.R covers type 11 across
#     cost x bias x svr_eps x dataset, but svr_eps (param.p, the SVR
#     epsilon-insensitive-tube loss margin) is a DIFFERENT parameter from
#     epsilon/param.eps (the Newton solver's own gradient-based stopping
#     criterion). That file holds epsilon at its default throughout and
#     never varies it.
#   - test-reference-coefficients.R pins type 11 at a single
#     (cost=1, default epsilon) point only.
# This file closes that gap: type 11 W pinned across a real
# epsilon x cost x dataset grid, so a Newton-CG-rewrite-specific drift
# that only shows up at a loose or tight stopping tolerance cannot hide
# behind the single default-epsilon point covered elsewhere.
#
# Types 12 and 13 (L2R_L2LOSS_SVR_DUAL, L2R_L1LOSS_SVR_DUAL) are included
# at the exact same grid points as a CONTROL group, not as required
# coverage: these are dual solvers, confirmed by direct source reading to
# be untouched by the tron->newton rewrite (bit-for-bit unaffected against
# pre-upgrade goldens, verified separately). If a future regression run
# ever shows type 11 moving while 12/13 stay pinned at this file's values,
# that isolates the cause to the newton.cpp rewrite (or something type-11
# specific) rather than a shared upstream/glue change.
#
# NOTE ON "the default": R/LiblineaR.R's @param epsilon docs and
# src/trainLinear.c's setup_params() (the `param.eps <= 0` switch, case
# L2R_L2LOSS_SVR) agree that type 11's actual built-in default epsilon is
# 0.001, NOT 0.01 (0.01 is the primal classification default, types
# 0/2/5/6). The grid below still includes 0.01 as a real, commonly-used
# value worth pinning, but does not call it "the default" for type 11.
# The actual default (0.001) is pinned separately, below, via an
# epsilon=NULL-vs-explicit-0.001 invariant check -- the same pattern
# test-vendor-golden-classweight-epsilon-bias.R uses to freeze that
# invariant for types 0/1/2/3 (the exact invariant the 2.10-25
# epsilon-default bug broke).
#
# Fixture provenance: tests/testthat/fixtures/vendor-golden-type11-
# epsilon.rds is produced by tests/testthat/fixtures/generate-vendor-
# golden-type11-epsilon.R (see that script's header for full provenance
# and the regenerate-deliberately-not-silently policy). It was generated
# against the CURRENT, post-Stage-(ii) vendored LIBLINEAR build -- i.e.
# these are NOT pre-upgrade values, they are the freshly-established
# baseline for everything from this stage forward. Regenerate only once a
# further reviewed vendor-upgrade stage is meant to change these numbers.
#
# Dual solvers (types 12, 13) route randomness through R's own RNG
# (rand_between() in src/linear.cpp), so set.seed() is called immediately
# before EVERY LiblineaR() call below (type 11 is primal and doesn't
# strictly need it, but is seeded too for uniformity with 12/13 and to
# match the generator script's loop exactly), not once at the top of the
# file.
#
# Tolerance: re-running this exact grid fresh against the just-generated
# fixture (same build, same machine, done manually before writing this
# file) reproduced every one of the 94 pinned values bit-for-bit
# (max(abs(fresh - frozen)) == 0 across the whole grid, including at
# epsilon=1e-8 with cost=100, the numerically stiffest corner). Observed
# |W| magnitudes in this fixture range from 0 to ~2.61. 1e-8 is used
# below: tight enough to catch a real regression against that
# bit-exact baseline, matching the tolerance already used for the sibling
# type-11 grid in test-vendor-golden-svr-paramgrid.R, while still
# absorbing negligible platform-level floating-point noise that isn't
# itself a solver-behavior change.

fixture_regr_correlated <- function() {
  set.seed(606)
  n <- 80
  p <- 6
  latent <- rnorm(n)
  x <- matrix(rnorm(n * p, sd = 0.5), ncol = p) + latent
  colnames(x) <- paste0("x", 1:p)
  y <- as.vector(x %*% c(2, -1, 0.5, 0, 1.5, -0.5)) + rnorm(n, sd = 1)
  list(x = x, y = y)
}

golden <- readRDS(test_path("fixtures", "vendor-golden-type11-epsilon.rds"))

datasets <- list(
  regr = fixture_regr(),
  corr = fixture_regr_correlated()
)

epsilon_list <- list(
  e1em8 = 1e-8,
  e1em4 = 1e-4,
  e0.01 = 0.01,
  e0.1  = 0.1,
  e1    = 1
)
costs <- c(0.01, 1, 100)

tol <- 1e-8

test_that("type 11 (L2R_L2LOSS_SVR primal) W matches frozen golden grid across epsilon x cost, two datasets", {
  # Risk covered (REQUIRED coverage): the tron->newton rewrite
  # (src/tron.cpp -> src/newton.cpp) changed HOW the primal SVR solver
  # decides it has converged. Any change to that stopping test, to the
  # Armijo line-search step, or to the Newton-CG inner loop that shifts
  # converged coefficients at any point in this epsilon x cost x dataset
  # grid -- not just the single default-epsilon point covered elsewhere --
  # is caught here.
  t <- 11
  for (ds_name in names(datasets)) {
    ds <- datasets[[ds_name]]
    for (eps_name in names(epsilon_list)) {
      for (co in costs) {
        key <- paste0("type", t, "_eps", eps_name, "_cost", co, "_", ds_name)
        set.seed(999)
        m <- suppressWarnings(
          LiblineaR(ds$x, ds$y, type = t, cost = co,
                    epsilon = epsilon_list[[eps_name]])
        )
        expect_equal(unname(m$W), golden[[key]], tolerance = tol, label = key)
      }
    }
  }
})

test_that("CONTROL: type 12 (L2R_L2LOSS_SVR_DUAL) W matches frozen golden grid at the same epsilon x cost points", {
  # Risk covered: this is a diff baseline, not a coverage requirement --
  # type 12 is a dual solver, confirmed unaffected by the tron->newton
  # rewrite (bit-for-bit vs. pre-upgrade goldens, verified separately). If
  # this ever starts failing while the type-11 test above still passes (or
  # vice versa), that isolates whether a future regression is specific to
  # the Newton-CG primal path or shared across the primal/dual boundary.
  t <- 12
  for (ds_name in names(datasets)) {
    ds <- datasets[[ds_name]]
    for (eps_name in names(epsilon_list)) {
      for (co in costs) {
        key <- paste0("type", t, "_eps", eps_name, "_cost", co, "_", ds_name)
        set.seed(999)
        m <- suppressWarnings(
          LiblineaR(ds$x, ds$y, type = t, cost = co,
                    epsilon = epsilon_list[[eps_name]])
        )
        expect_equal(unname(m$W), golden[[key]], tolerance = tol, label = key)
      }
    }
  }
})

test_that("CONTROL: type 13 (L2R_L1LOSS_SVR_DUAL) W matches frozen golden grid at the same epsilon x cost points", {
  # Risk covered: same control rationale as type 12 above, on the
  # hinge-loss (L1) dual SVR variant -- a different code path
  # (solve_l2r_l1l2_svr in src/linear.cpp) from type 12's squared-loss
  # dual, also confirmed unaffected by this stage.
  t <- 13
  for (ds_name in names(datasets)) {
    ds <- datasets[[ds_name]]
    for (eps_name in names(epsilon_list)) {
      for (co in costs) {
        key <- paste0("type", t, "_eps", eps_name, "_cost", co, "_", ds_name)
        set.seed(999)
        m <- suppressWarnings(
          LiblineaR(ds$x, ds$y, type = t, cost = co,
                    epsilon = epsilon_list[[eps_name]])
        )
        expect_equal(unname(m$W), golden[[key]], tolerance = tol, label = key)
      }
    }
  }
})

test_that("epsilon = NULL and epsilon = 0.001 (type 11's actual default) give IDENTICAL W, both datasets", {
  # Risk covered: this is the exact invariant the 2.10-25 epsilon-default
  # bug broke for other types (a wrong hardcoded default in the C glue
  # diverging from what NULL actually resolved to) -- pinned here for type
  # 11 specifically, at the correct default (0.001 per src/trainLinear.c's
  # setup_params(), NOT 0.01 -- see file header). If a future vendor-
  # upgrade stage changes type 11's internal default epsilon, this must be
  # updated deliberately, in lockstep with R/LiblineaR.R's @param epsilon
  # documentation, not left to drift silently.
  for (ds_name in names(datasets)) {
    ds <- datasets[[ds_name]]

    set.seed(999)
    m_null <- suppressWarnings(LiblineaR(ds$x, ds$y, type = 11, cost = 1, epsilon = NULL))
    set.seed(999)
    m_explicit <- suppressWarnings(LiblineaR(ds$x, ds$y, type = 11, cost = 1, epsilon = 0.001))

    key_null <- paste0("defaultcheck_type11_null_", ds_name)
    key_explicit <- paste0("defaultcheck_type11_explicit_", ds_name)

    # Freeze against the golden fixture (drift vs. the verified-correct,
    # post-Stage-(ii) solver)...
    expect_equal(unname(m_null$W), golden[[key_null]], tolerance = tol,
                 label = paste0(ds_name, ": epsilon=NULL vs golden"))
    expect_equal(unname(m_explicit$W), golden[[key_explicit]], tolerance = tol,
                 label = paste0(ds_name, ": epsilon=0.001 vs golden"))

    # ...and directly against each other (the actual invariant under
    # test): NULL must resolve to exactly the true default, bit-for-bit.
    expect_equal(unname(m_null$W), unname(m_explicit$W), tolerance = 0,
                 label = paste0(ds_name, ": NULL == explicit epsilon=0.001"))
  }
})
