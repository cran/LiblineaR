# Guards the golden-value portability policy itself
# (helper-golden-portability.R). The golden fixtures across this suite rely
# on that policy to decide what may be asserted on CRAN; if its machinery
# silently broke, every one of those files would quietly loosen or tighten
# without anyone noticing. These tests are cheap and run everywhere.

# withr is not a declared dependency of this package, so NOT_CRAN is set and
# restored here by hand, using withr's own technique: register the restore
# as an on.exit() in the CALLER's frame, which testthat unwinds at the end
# of the enclosing test_that() block. Leaking NOT_CRAN would silently change
# what every later test file asserts, since the exact golden pins are gated
# on it (see golden_exact_pin_enabled()).
# expect_golden() deliberately emits SEVERAL expectations per call (shape,
# length, finiteness, and the exact pin off CRAN), so testthat's own
# single-expectation meta-expectations cannot be used to test it: they
# require exactly one success or failure and would report "expected 1
# success" against a call that legitimately produces three. These capture
# instead whether the call failed AT ALL. The inner failure is caught here
# rather than by testthat's reporter, so it is not recorded as a failure of
# this file.
golden_fails <- function(code) {
  tryCatch({
    code
    FALSE
  }, expectation_failure = function(cnd) TRUE)
}
expect_golden_passes <- function(code) expect_false(golden_fails(code))
expect_golden_fails  <- function(code) expect_true(golden_fails(code))

local_not_cran <- function(value, envir = parent.frame()) {
  old <- Sys.getenv("NOT_CRAN", unset = NA_character_)
  restore <- if (is.na(old)) {
    quote(Sys.unsetenv("NOT_CRAN"))
  } else {
    bquote(Sys.setenv(NOT_CRAN = .(old)))
  }
  Sys.setenv(NOT_CRAN = value)
  do.call(base::on.exit, list(restore, add = TRUE), envir = envir)
  invisible(NULL)
}

test_that("solver_state() classifies LIBLINEAR's verbose log correctly", {
  # Risk covered: an upstream wording change in LIBLINEAR's info() messages
  # (these strings are matched literally) would make every capped fit look
  # converged, so the suite would start pinning truncated iterates exactly
  # on CRAN again -- the precise failure that broke the 2.50-1 submission.
  expect_equal(solver_state("optimization finished, #iter = 12"), "converged")
  expect_equal(solver_state(character(0)), "converged")

  expect_equal(solver_state(c("optimization finished, #iter = 300",
                              "WARNING: reaching max number of iterations")),
               "capped_truncated")

  # A switching type logs BOTH the cap warning and the switch; the switch
  # must win, since the returned W then comes from the primal solver.
  expect_equal(solver_state(c("WARNING: reaching max number of iterations",
                              "Switching to use -s 2")),
               "capped_refit")

  expect_equal(solver_state("WARNING: reaching max number of Newton iterations"),
               "capped_newton")

  # "...max number of Newton iterations" must not be mistaken for the dual
  # cap message, which is a different solver and a different risk.
  expect_false(identical(
    solver_state("WARNING: reaching max number of Newton iterations"),
    "capped_truncated"))

  expect_false(golden_state_is_capped("converged"))
  expect_true(golden_state_is_capped("capped_truncated"))
  expect_true(golden_state_is_capped("capped_refit"))
})

test_that("ll_golden_fit() reports state alongside the model without leaking output", {
  f <- fixture_classif()
  set.seed(42)
  out <- capture.output(g <- ll_golden_fit(data = f$x, target = f$y, type = 0))
  expect_length(out, 0)                       # verbose log captured, not printed
  expect_true(is.list(g) && !is.null(g$model$W))
  expect_true(g$state %in% c("converged", "capped_truncated",
                             "capped_refit", "capped_newton"))
  expect_true(length(g$log) > 0)
})

test_that("a fixture that is not stable under tightening asserts no value on CRAN", {
  # Risk covered: the 2.50-1 arm64 ERROR. When `epsilon` leaves a stopping
  # ball wider than the pin tolerance, WHICH point of that ball a platform
  # returns is a property of its arithmetic, so ANY numeric assertion on it
  # -- a loose pointwise tolerance or a solution-magnitude band, both of
  # which this file used to allow -- can fail somewhere on the build farm.
  # On arm64 one fixture's magnitude alone moved 10.7%. The only safe rule
  # is that nothing numeric survives, and this test holds that rule in place
  # against a well-meaning future re-loosening.
  local_not_cran("false")
  unstable <- list(state = "converged", tighten_fit = list(state = "converged", tighten_move = 1))

  # Wildly wrong values, same shape: must pass, because on CRAN an unstable
  # fixture is not compared pointwise at all.
  expect_golden_passes(
    expect_golden(matrix(c(1e6, -1e6), 1), matrix(c(0.1, 0.2), 1),
                  key = "fake", fit = unstable))

  # The invariants that DO remain must still bite.
  expect_golden_fails(
    expect_golden(matrix(c(1, 2, 3), 1), matrix(c(0.1, 0.2), 1),
                  key = "fake", fit = unstable))
  expect_golden_fails(
    expect_golden(matrix(c(NaN, 1), 1), matrix(c(0.1, 0.2), 1),
                  key = "fake", fit = unstable))
  expect_golden_fails(
    expect_golden(matrix(c(Inf, 1), 1), matrix(c(0.1, 0.2), 1),
                  key = "fake", fit = unstable))
})

test_that("a fixture that is stable under tightening is still pinned exactly on CRAN", {
  # Risk covered: the mirror image -- a change that loosens EVERY fixture
  # instead of only the unstable ones would hide a real vendor regression
  # across the whole build farm.
  local_not_cran("false")
  stable <- list(state = "converged", tighten_move = 0)

  expect_golden_passes(
    expect_golden(matrix(c(0.1, 0.2), 1), matrix(c(0.1, 0.2), 1),
                  key = "fake", fit = stable))
  expect_golden_fails(
    expect_golden(matrix(c(0.1, 0.9), 1), matrix(c(0.1, 0.2), 1),
                  key = "fake", fit = stable))
})

test_that("the portability verdict is taken against the caller's own pin tolerance", {
  # Risk covered: a fixed internal threshold. A file that pins at 1e-8 (the
  # SVR grids) needs a stricter stability bar than one pinning at 1e-6, so
  # the stability bar must be exact_tol itself, not a constant.
  local_not_cran("false")
  a <- matrix(c(1, 2), 1); b <- matrix(c(1.5, 2.5), 1)

  # fit = list(state = "converged", tighten_move = 1e-7): portable at a 1e-6 pin, not at a 1e-8 pin.
  expect_golden_fails(
    expect_golden(a, b, key = "fake", fit = list(state = "converged", tighten_move = 1e-7), exact_tol = 1e-6))
  expect_golden_passes(
    expect_golden(a, b, key = "fake", fit = list(state = "converged", tighten_move = 1e-7), exact_tol = 1e-8))
})

test_that("an unmeasurable probe is treated as not portable", {
  # Risk covered: .golden_tighten_move() returning Inf because the tightened
  # re-solve errored or returned a different shape. Failing open there would
  # pin a value nobody checked.
  local_not_cran("false")
  expect_golden_passes(
    expect_golden(matrix(c(1e6, -1e6), 1), matrix(c(0.1, 0.2), 1),
                  key = "fake", fit = list(state = "converged", tighten_move = Inf)))

  expect_equal(.golden_move(c(1, 2), c(1, 2, 3)), Inf)
  expect_equal(.golden_move(c(1, NA), c(1, 2)), Inf)
  expect_equal(.golden_move(c(1, 2), c(0, 0)), 2)      # zero-scale reference
  expect_equal(.golden_move(c(1, 2), c(1, 2)), 0)
})

test_that("the exact pin on an unstable fixture is restored off CRAN", {
  # Risk covered: "moved off the build farm" silently becoming "deleted".
  # The wildly wrong value accepted on the CRAN path above must fail here.
  local_not_cran("true")

  expect_golden_fails(
    expect_golden(matrix(c(1e6, -1e6), 1), matrix(c(0.1, 0.2), 1),
                  key = "fake", fit = list(state = "converged", tighten_move = 1)))
  expect_golden_passes(
    expect_golden(matrix(c(0.1, 0.2), 1), matrix(c(0.1, 0.2), 1),
                  key = "fake", fit = list(state = "converged", tighten_move = 1)))
})

test_that("the tightening probe leaves the RNG stream where the main fit left it", {
  # Risk covered: the probe re-solving from the caller's RNG state and not
  # restoring it. Every golden fixture re-seeds before each fit, but a
  # fixture that did not would silently change its own values the day the
  # probe was added -- and the whole suite's golden values would have to be
  # regenerated for no reason.
  f <- fixture_classif()
  set.seed(42); g <- ll_golden_fit(data = f$x, target = f$y, type = 3)
  after_probe <- .Random.seed

  set.seed(42)
  invisible(capture.output(
    suppressWarnings(LiblineaR(f$x, f$y, type = 3, verbose = TRUE))))
  expect_identical(after_probe, .Random.seed)
  expect_true(is.finite(g$tighten_move) || is.infinite(g$tighten_move))
})

test_that("the probe separates default-epsilon from tight-epsilon fits", {
  # Risk covered: a probe that reports everything portable (which would
  # restore the 2.50-1 failure) or nothing portable (which would silently
  # delete CRAN's golden coverage). Uses the very scenario that broke arm64:
  # type 3 at default epsilon sits in a wide stopping ball, and the same
  # problem at 1e-6 does not.
  f <- fixture_classif()
  set.seed(42)
  loose <- ll_golden_fit(data = f$x, target = f$y, type = 3, cost = 1, bias = 5)
  set.seed(42)
  tight <- ll_golden_fit(data = f$x, target = f$y, type = 3, cost = 1, bias = 5,
                         epsilon = 1e-6)

  expect_gt(loose$tighten_move, GOLDEN_EXACT_TOL)
  expect_lte(tight$tighten_move, GOLDEN_EXACT_TOL)
})

test_that("a capped-truncated fit is never pinned on CRAN, however stable it looks", {
  # Risk covered: the blind spot that broke the FIRST attempt at this
  # policy. Tightening epsilon on a fit that stopped at its iteration cap
  # just hits the same cap after the same iterations, so the probe reports
  # ZERO movement for a value that is pure arithmetic-path artefact. Two
  # type-3 multiclass fixtures at epsilon=1e-6 did exactly this and still
  # differed under an FMA build. The state axis is what catches them.
  local_not_cran("false")
  for (st in c("capped_truncated", "capped_newton")) {
    expect_false(golden_is_portable(list(state = st, tighten_move = 0)),
                 label = st)
    expect_golden_passes(
      expect_golden(matrix(c(1e6, -1e6), 1), matrix(c(0.1, 0.2), 1),
                    key = "fake", fit = list(state = st, tighten_move = 0)))
  }
  # capped_refit is the exception: the cap was hit, then the PRIMAL solver
  # re-solved and the returned W is that converged result.
  expect_true(golden_is_portable(list(state = "capped_refit", tighten_move = 0)))
  expect_false(golden_is_portable(list(state = "converged", tighten_move = 1)))
  # A malformed fit must not be read as portable.
  expect_false(golden_is_portable(list()))
})

test_that("exact same-platform golden pins are asserted off CRAN only", {
  # Deliberately visible: this records a SKIP in CRAN's own check log, so a
  # reader of that log can see that the exact pins exist and were withheld
  # here on purpose, rather than silently never running.
  skip_on_cran()
  expect_true(golden_exact_pin_enabled())
})
