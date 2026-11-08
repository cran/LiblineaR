# Generator for tests/testthat/fixtures/vendor-golden-type11-epsilon.rds
#
# Plain R script -- NOT a test file, never auto-run by testthat. Run it
# deliberately, from the package root, with:
#   Rscript tests/testthat/fixtures/generate-vendor-golden-type11-epsilon.R
#
# WHAT IT CAPTURES AND WHY
# ------------------------
# Stage (ii) of the vendor upgrade (v2.30 -> v2.40) replaced upstream's
# TRON trust-region solver with a Newton-CG + Armijo line-search solver
# (src/tron.cpp -> src/newton.cpp) for the three PRIMAL solver types
# 0 (L2R_LR), 2 (L2R_L2LOSS_SVC) and 11 (L2R_L2LOSS_SVR). That rewrite
# changes what "converged" means step by step, so the termination
# tolerance (epsilon / param.eps) -- which is exactly what the new solver's
# stopping test consumes -- is the single parameter most likely to expose a
# behavioural difference between the old TRON and the new Newton-CG code
# path.
#
# Despite that, until this file existed there was NO golden coverage of
# epsilon sensitivity for type 11 anywhere in this battery:
#   - test-vendor-golden-classweight-epsilon-bias.R varies epsilon, but
#     only for types 0, 1, 2, 3 (classification) -- type 11 is absent.
#   - test-vendor-golden-svr-paramgrid.R covers type 11 across
#     cost x bias x svr_eps x dataset, but svr_eps is the SVR
#     epsilon-insensitive-tube loss margin (param.p), a completely
#     different parameter from epsilon/param.eps, which controls the
#     Newton solver's own gradient-based stopping criterion. svr_eps is
#     held at its default throughout that grid; epsilon is never varied
#     there at all.
#   - test-reference-coefficients.R pins type 11 at a single
#     (cost=1, default epsilon) point only.
# This file closes exactly that gap: type 11 (L2R_L2LOSS_SVR primal) W
# coefficients pinned across a real epsilon x cost x dataset grid, so a
# Newton-CG-rewrite-specific drift that only shows up at a loose or tight
# stopping tolerance cannot hide behind the single default-epsilon point
# that other files already cover.
#
# THE GRID
# --------
#   type    = 11 (required coverage), plus 12 and 13 (L2R_L2LOSS_SVR_DUAL,
#             L2R_L1LOSS_SVR_DUAL) as a control group: these are DUAL
#             solvers, confirmed by direct source reading to be untouched
#             by the tron->newton rewrite (bit-for-bit unaffected, verified
#             separately against pre-upgrade goldens). Including them here,
#             at the exact same grid points, gives a same-file diff
#             baseline: if a future regression run shows type 11 moving
#             while 12/13 stay frozen at THIS file's own values, that is
#             strong evidence the newton.cpp rewrite (or something type-11-
#             specific) is the cause, not a shared upstream/glue change.
#   epsilon = 1e-8, 1e-4, 0.01, 0.1, 1
#             NOTE ON "the default": R/LiblineaR.R's @param epsilon docs
#             and src/trainLinear.c's setup_params() (param.eps <= 0
#             switch, case L2R_L2LOSS_SVR) agree that type 11's actual
#             built-in default is 0.001, NOT 0.01 -- 0.01 is the primal
#             classification default (types 0/2/5/6). This grid still
#             includes 0.01 (a real, commonly-used value worth pinning
#             regardless), but the "(the default)" label from the task
#             this file was written for is factually wrong for type 11 and
#             is corrected here. The actual default (0.001) is pinned
#             separately below via the epsilon=NULL-vs-explicit-0.001
#             invariant check, mirroring the pattern in
#             test-vendor-golden-classweight-epsilon-bias.R.
#   cost    = 0.01, 1, 100
#   dataset = fixture_regr() (from helper-fixtures.R: small, nearly-linear,
#             nearly-noiseless -- n=40, 3 features) and
#             fixture_regr_correlated() (defined in this script: n=80, 6
#             correlated features via a shared latent factor, noisier
#             target -- see below for why).
#
# fixture_regr_correlated(): fixture_regr()'s columns are independent and
# the fit is near-exact (residual sd ~ 0.1 against a signal sd of several
# units), so the Newton solver converges in very few steps regardless of
# epsilon -- a harder problem is needed to make epsilon actually change the
# number of iterations, and therefore to make an epsilon-sensitivity gap
# worth pinning. This second dataset makes each feature column a shared
# latent factor plus independent noise (x[, j] = latent + rnorm(n, sd=0.5)
# for j=1..6), producing genuine multicollinearity, plus target noise on
# the same order as the signal -- both of which lengthen the optimization
# path and make the stopping tolerance actually bite. Own explicit seed
# (606), independent of fixture_regr()'s internal seed (321) and of
# fixture_regr_highnoise()'s seed (777) in test-vendor-golden-svr-
# paramgrid.R.
#
# These are regression-baseline values pinned from the CURRENT (post
# Stage-(ii)-merge) vendored LIBLINEAR build, not externally-audited ground
# truth (LIBLINEAR ships no independent reference harness to diff
# against) -- consistent with every other vendor-golden-*.rds fixture in
# this battery. Regenerate this .rds deliberately (never silently, never
# as a side effect of a failing test), only once a reviewed vendor-upgrade
# stage (e.g. a later v2.40 -> v2.50 step touching newton.cpp again) is
# meant to change these values, and only after confirming the new values
# by other means.
#
# HOW TO RUN
# ----------
# From the package root (working directory = package root), with the
# package already installed (library(LiblineaR) -- this script does not
# install it):
#   Rscript tests/testthat/fixtures/generate-vendor-golden-type11-epsilon.R
#
# Dual solvers (types 12, 13) route randomness through R's own RNG
# (rand_between() in src/linear.cpp), so set.seed() is called immediately
# before EVERY LiblineaR() call below (type 11 is primal and doesn't
# strictly need it, but is seeded too for uniformity with types 12/13 and
# so the loop body needs no type-conditional), matching the discipline
# already used throughout this battery.

library(LiblineaR)

# This script is run directly with Rscript (cwd = package root), so
# source the fixture helper directly rather than relying on testthat's
# auto-loading of helper-*.R (which only happens inside a testthat run).
source(file.path("tests", "testthat", "helper-fixtures.R"))

# Second regression dataset: correlated features + non-trivial noise, so
# the Newton solver's iteration count -- and hence sensitivity to the
# epsilon stopping tolerance -- is non-trivial (see header comment above).
# n=80, 6 features. Own explicit seed (606), local to this script (and
# duplicated verbatim in the test file, matching the existing
# fixture_regr_highnoise() pattern in test-vendor-golden-svr-paramgrid.R).
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

datasets <- list(
  regr = fixture_regr(),
  corr = fixture_regr_correlated()
)

types <- c(11, 12, 13)
epsilon_list <- list(
  e1em8 = 1e-8,
  e1em4 = 1e-4,
  e0.01 = 0.01,
  e0.1  = 0.1,
  e1    = 1
)
costs <- c(0.01, 1, 100)

golden <- list()

for (ds_name in names(datasets)) {
  ds <- datasets[[ds_name]]
  for (t in types) {
    for (eps_name in names(epsilon_list)) {
      for (co in costs) {
        key <- paste0("type", t, "_eps", eps_name, "_cost", co, "_", ds_name)
        set.seed(999)
        m <- suppressWarnings(
          LiblineaR(ds$x, ds$y, type = t, cost = co,
                    epsilon = epsilon_list[[eps_name]])
        )
        golden[[key]] <- unname(m$W)
      }
    }
  }
}

# --- epsilon=NULL vs explicit true default (0.001) invariant, type 11 ----
# Mirrors the pattern in generate-vendor-golden-classweight-epsilon-bias.R:
# freezes that epsilon=NULL resolves to EXACTLY the documented/source-code
# default for type 11 (0.001, per src/trainLinear.c's setup_params(),
# case L2R_L2LOSS_SVR -- see header comment above), bit-for-bit, on both
# datasets.
for (ds_name in names(datasets)) {
  ds <- datasets[[ds_name]]

  set.seed(999)
  m_null <- suppressWarnings(LiblineaR(ds$x, ds$y, type = 11, cost = 1, epsilon = NULL))
  golden[[paste0("defaultcheck_type11_null_", ds_name)]] <- unname(m_null$W)

  set.seed(999)
  m_explicit <- suppressWarnings(LiblineaR(ds$x, ds$y, type = 11, cost = 1, epsilon = 0.001))
  golden[[paste0("defaultcheck_type11_explicit_", ds_name)]] <- unname(m_explicit$W)
}

out_path <- file.path(
  "tests", "testthat", "fixtures", "vendor-golden-type11-epsilon.rds"
)
# version = 2 explicitly: the default serialize version 3 silently adds an
# undeclared R (>= 3.5.0) dependency, caught by R CMD build earlier in this
# project -- every fixture in this battery pins version = 2 for the same
# reason.
saveRDS(golden, out_path, version = 2)
cat("Wrote", out_path, "\n")
cat(length(golden), "golden W matrices\n")
