# VENDOR-UPGRADE GOLDEN BATTERY: cross-validation (cross=k) and findC=TRUE
# exact values.
#
# This is the safety net for the multi-stage vendor upgrade of the vendored
# LIBLINEAR C/C++ source (currently ~v2.1, going to v2.50), which will
# touch src/linear.cpp, src/tron.cpp (-> newton.cpp), trainLinear.c and
# predictLinear.c. test-cross-validation.R and test-findC.R only check
# properties (is it a number, is it in range, is it reproducible) -- this
# file pins the EXACT numeric values returned by the current,
# already-verified-correct solver so that any post-upgrade drift, expected
# (e.g. the v2.40 Newton-solver rewrite legitimately moving converged
# values, within tolerance) or a genuine regression, is caught by a
# failing assertion rather than silently passing through.
#
# Values are frozen in tests/testthat/fixtures/vendor-golden-cv-findc.rds,
# produced by tests/testthat/fixtures/generate-vendor-golden-cv-findc.R
# (see that script's header for full provenance and regeneration policy).
# Regenerate deliberately, never automatically, and only once a reviewed
# vendor-upgrade stage is meant to change these numbers.
#
# Tolerance: repeating the exact same grid twice in this session (same
# build, same machine) reproduced every value bit-for-bit (max|diff| == 0
# for cross_classif, cross_regr and findC alike -- verified manually before
# writing this file). 1e-6 is used below, matching the tolerance already
# used for golden-value pinning elsewhere in this project
# (test-reference-coefficients.R, test-sparse-dense-equivalence.R): tight
# enough to catch a real regression, loose enough to absorb negligible
# platform-level floating-point non-determinism that isn't itself a
# solver-behavior change.
#
# Dual solvers (types 1,3,4,7,12,13) route cross-validation fold-assignment
# randomness through R's own RNG (rand_between() in src/linear.cpp), so
# set.seed(777) -- matching the generator script -- is called immediately
# before EVERY LiblineaR() call below, never once at the top of the file.

fc <- fixture_classif()
fr <- fixture_regr()
golden <- readRDS(test_path("fixtures", "vendor-golden-cv-findc.rds"))

# --- cross=k classification accuracy ---------------------------------

test_that("cross=k classification accuracy matches frozen golden values, types 0-7, k in {3,5,10}", {
  # Risk covered: any change to the primal/dual training loop (linear.cpp),
  # the convergence criterion (tron.cpp -> newton.cpp), or the
  # cross-validation fold-building/scoring logic (trainLinear.c's
  # do_cross_validation) that shifts predicted labels on held-out folds,
  # even slightly, changes this accuracy number.
  for (t in 0:7) {
    for (k in c(3, 5, 10)) {
      key <- paste0("type", t, "_k", k)
      set.seed(777)
      acc <- LiblineaR(fc$x, fc$y, type = t, cross = k)
      expect_equal(acc, golden$cross_classif[[key]], tolerance = 1e-6,
                   label = paste("type", t, "k", k))
    }
  }
})

# --- cross=k regression MSE -------------------------------------------

test_that("cross=k regression MSE matches frozen golden values, types 11-13, k in {3,5,10}", {
  # Risk covered: same as above, but on the regression (SVR) code path --
  # catches drift in the epsilon-insensitive loss solvers (L2R_L2LOSS_SVR,
  # L2R_L1LOSS_SVR_DUAL, L2R_L2LOSS_SVR_DUAL) or in the MSE aggregation
  # across folds in trainLinear.c's cross_validation() SVR branch.
  for (t in 11:13) {
    for (k in c(3, 5, 10)) {
      key <- paste0("type", t, "_k", k)
      set.seed(777)
      mse <- suppressWarnings(LiblineaR(fr$x, fr$y, type = t, cross = k, svr_eps = 0.01))
      expect_equal(mse, golden$cross_regr[[key]], tolerance = 1e-6,
                   label = paste("type", t, "k", k))
    }
  }
})

# --- findC=TRUE discovered C -------------------------------------------

test_that("findC=TRUE discovered C matches frozen golden values, types 0 and 2, useInitC x cost x cross grid", {
  # Risk covered: upstream's warm-start parameter search (find_parameter_C
  # in linear.cpp, driven from trainLinear.c) doubling C from a start value
  # until cross-validation accuracy stops improving. A change to the
  # doubling schedule, the warm-start reuse of previous solutions, or the
  # underlying solver's convergence would move the discovered C, the
  # accuracy it's chosen on, or both.
  for (t in c(0, 2)) {
    for (useInitC in c(TRUE, FALSE)) {
      for (cost0 in c(1, 1000)) {
        for (k in c(3, 5)) {
          key <- paste0("type", t, "_useInitC", useInitC, "_cost", cost0, "_k", k)
          set.seed(777)
          bestC <- LiblineaR(fc$x, fc$y, type = t, cost = cost0,
                              findC = TRUE, useInitC = useInitC, cross = k)
          expect_equal(bestC, golden$findC[[key]], tolerance = 1e-6,
                       label = key)
        }
      }
    }
  }
})

# --- Former CANARY: findC=TRUE on type 11 (regression) used to error ---
#
# This tripwire (findC=TRUE on type 11 raises an error) existed from before
# the Phase A/B split through Stage (iii); Phase B (see
# VENDOR-UPGRADE-PLAN.md) is what actually implements -C/findC support for
# SVR (L2R_L2LOSS_SVR), superseding it. See
# tests/testthat/test-vendor-golden-findc-svr.R for the real feature
# coverage. Kept as a narrower, still-true fact: findC is still rejected
# for every OTHER non-whitelisted type.

test_that("findC=TRUE is still rejected for types other than 0, 2 and 11", {
  expect_error(
    suppressWarnings(LiblineaR(fc$x, fc$y, type = 1, findC = TRUE, cross = 3)),
    "Warm-start parameter search only available for L2R_LR, L2R_L2LOSS_SVC and L2R_L2LOSS_SVR",
    fixed = TRUE
  )
  expect_error(
    suppressWarnings(LiblineaR(fr$x, fr$y, type = 12, findC = TRUE, cross = 3)),
    "Warm-start parameter search only available for L2R_LR, L2R_L2LOSS_SVC and L2R_L2LOSS_SVR",
    fixed = TRUE
  )
})

# --- Former CANARY block: ONECLASS_SVM=21 reachability ---
#
# Stage (ii) [v2.30->v2.40]'s risk analysis added 3 canaries here pinning
# that type=21 (ONECLASS_SVM) stayed unreachable from every angle, each
# explicitly designed to start FAILING the moment a future phase wired it
# up end to end -- Phase B (see VENDOR-UPGRADE-PLAN.md) is that moment.
# Two of the three canaries were removed here (their job was exactly this:
# mark when R-level support arrived, so real coverage could replace them
# in the same reviewed change) -- see tests/testthat/test-vendor-golden-
# oneclass-svm.R for the actual Phase B feature coverage that supersedes
# them. The third is kept below: it tests something that remains true
# regardless of type=21's R-level reachability (internal C++ symbols
# should never be directly .C()-callable), so it's still real coverage,
# just reframed.

test_that("R_useDynamicSymbols(dll, FALSE) (src/init.c) blocks calling internal native symbols (e.g. solve_oneclass_svm()) directly by name", {
  # Risk covered: src/init.c registers exactly two routines (predictLinear,
  # trainLinear) and calls R_useDynamicSymbols(dll, FALSE), so any other
  # native symbol -- including solve_oneclass_svm(), an internal function in
  # linear.cpp -- must remain unreachable via .C() by name, even though it
  # exists in the compiled object and is now reachable through the public
  # API (LiblineaR(type=21)). This guards against an accidental future
  # registration change (a new CEntries row) or a flipped
  # R_useDynamicSymbols flag silently exposing internal solver entry points
  # as part of the package's native API.
  #
  # Exact error text verified manually before writing this regex (R's own
  # wording differs by call form/platform, so it is not guessed):
  #   .C("solve_oneclass_svm", PACKAGE = "LiblineaR") ->
  #   "\"solve_oneclass_svm\" not available for .C() for package \"LiblineaR\""
  expect_error(
    .C("solve_oneclass_svm", PACKAGE = "LiblineaR"),
    "not available for \\.C\\(\\)"
  )
})
