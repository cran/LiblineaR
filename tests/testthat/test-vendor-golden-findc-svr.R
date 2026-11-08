# VENDOR-UPGRADE GOLDEN BATTERY: findC extended to SVR (type 11), Phase B
# (see VENDOR-UPGRADE-PLAN.md).
#
# Upstream v2.30 (already vendored since Stage (i)) renamed/restructured
# find_parameter_C() into find_parameters(), adding a joint C/svr_eps
# search for L2R_L2LOSS_SVR (type 11) alongside the original C-only search
# for L2R_LR/L2R_L2LOSS_SVC (types 0/2). LiblineaR() only exposed the
# types-0/2 path until now; Phase B extends findC=TRUE to type 11 too,
# reusing 'useInitC'/'svr_eps' as the search anchor for svr_eps (rather
# than a new, near-duplicate argument) -- see src/trainLinear.c's
# do_find_parameter_C() and setup_params().
#
# Design note (checked directly, not assumed): unlike start_C (which sets
# a true lower bound the C-doubling search grows from), start_p sets an
# UPPER anchor for a descending svr_eps sweep that always continues down
# to svr_eps=0 and keeps whichever value minimizes CV error -- so a small
# svr_eps relative to the scale of the regression target can (and does,
# for fixture_regr()) collapse the sweep to effectively just testing
# svr_eps=0. This is genuine upstream find_parameters() behavior
# (linear.cpp), not a bug in this package's exposure of it.
#
# Return shape: findC=TRUE for type 11 returns a named list
# list(C=..., svr_eps=...) instead of the plain scalar every other findC
# type returns, since there are now two "best" values to report. This is
# additive, not breaking: verified below that types 0/2 still return a
# plain scalar, unchanged.

test_that("findC=TRUE for type 11 returns list(C=..., svr_eps=...), not a plain scalar", {
  fr <- fixture_regr()
  set.seed(777)
  res <- suppressWarnings(LiblineaR(fr$x, fr$y, type = 11, findC = TRUE, cross = 3))

  expect_type(res, "list")
  expect_setequal(names(res), c("C", "svr_eps"))
  expect_true(is.numeric(res$C) && length(res$C) == 1 && !is.na(res$C) && res$C > 0)
  expect_true(is.numeric(res$svr_eps) && length(res$svr_eps) == 1 && !is.na(res$svr_eps) && res$svr_eps >= 0)
})

test_that("findC=TRUE for types 0 and 2 (pre-existing) still returns a plain scalar, unchanged", {
  # Risk covered: Phase B's findC changes must be additive for type 11
  # only, not alter the established, documented return contract for the
  # types that already had findC support.
  fc <- fixture_classif()
  for (type in c(0, 2)) {
    set.seed(777)
    res <- LiblineaR(fc$x, fc$y, type = type, findC = TRUE, cross = 3)
    expect_true(is.numeric(res) && length(res) == 1 && !is.na(res),
                label = paste("type", type))
  }
})

test_that("findC=TRUE for type 11 golden grid: useInitC TRUE/FALSE, both deterministic and reproducible", {
  # Golden-pinned (not just shape-checked): freezes the exact (C, svr_eps)
  # pair found on fixture_regr() for both useInitC settings, confirmed
  # reproducible via set.seed() like every other test in this battery
  # (find_parameters()'s own internal fold-shuffling uses rand_between(),
  # routed through R's RNG since Stage (i)).
  fr <- fixture_regr()

  set.seed(777)
  res_init <- suppressWarnings(LiblineaR(fr$x, fr$y, type = 11, findC = TRUE, cross = 3, useInitC = TRUE))
  set.seed(777)
  res_no_init <- suppressWarnings(LiblineaR(fr$x, fr$y, type = 11, findC = TRUE, cross = 3, useInitC = FALSE))

  # Reproducibility: same seed, same call -> identical result.
  set.seed(777)
  res_init_repeat <- suppressWarnings(LiblineaR(fr$x, fr$y, type = 11, findC = TRUE, cross = 3, useInitC = TRUE))
  expect_equal(res_init, res_init_repeat)

  # useInitC=TRUE (svr_eps defaults to 0.1, small relative to fixture_regr()'s
  # target scale) collapses the descending sweep to svr_eps=0, per the design
  # note above -- checked manually before writing this assertion.
  expect_equal(res_init$svr_eps, 0)
  expect_equal(res_init$C, 256)

  # useInitC=FALSE runs the full 20-step default sweep (anchored at
  # max(abs(target))) -- also checked manually to land on the same
  # svr_eps=0/C=256 optimum for this fixture (a wider search finding the
  # same optimum a narrower one already found is itself a useful
  # consistency signal, not a coincidence to paper over).
  expect_equal(res_no_init$svr_eps, 0)
  expect_equal(res_no_init$C, 256)
})

test_that("findC is still rejected for a type outside {0, 2, 11}, with an updated error message", {
  fc <- fixture_classif()
  expect_error(
    suppressWarnings(LiblineaR(fc$x, fc$y, type = 6, findC = TRUE, cross = 3)),
    "Warm-start parameter search only available for L2R_LR, L2R_L2LOSS_SVC and L2R_L2LOSS_SVR",
    fixed = TRUE
  )
})
