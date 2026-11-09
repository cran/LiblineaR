# ---------------------------------------------------------------------------
# Golden-value portability policy
# ---------------------------------------------------------------------------
#
# WHY THIS FILE EXISTS
#
# A golden value is only portable across CRAN's build farm if it pins a
# CONVERGED optimum. A converged optimum is a property of the optimization
# problem: every platform that solves the problem correctly finds it. A
# state reached after exactly N coordinate-descent sweeps, with the solver
# stopped at its iteration cap, is a property of the ARITHMETIC PATH, and
# different platforms do not share an arithmetic path.
#
# LiblineaR 2.50-1 shipped one such fixture by accident and CRAN's Windows
# pretest caught it (see VENDOR-UPGRADE-PLAN.md). The lesson is not "fix
# that fixture" -- it is that nothing in this suite NOTICED. This file is
# the mechanism that notices.
#
# Two distinct things can make a truncated iterate non-portable:
#
#   1. libm differences. Platform `exp`/`log` implementations differ in the
#      last ulp. Only solvers that call them mid-iteration are exposed:
#        solve_l2r_lr_dual()    (type 7)      -- 2x log() in the inner loop
#        solve_l2r_l1l2_svc()   (types 1, 3)  -- none
#        solve_l2r_l1l2_svr()   (types 12,13) -- none
#      This is what broke type 7 on Windows R-devel.
#
#   2. Arithmetic reordering. IEEE +,-,*,/ are exactly specified, so the
#      non-transcendental solvers ARE bit-identical across platforms GIVEN
#      THE SAME OPERATION ORDER. FMA contraction (on by default in gcc and
#      clang) changes that order, and differs between x86-64 and arm64.
#      CRAN builds for macOS arm64. A truncated iterate from types 1/3/12/13
#      is therefore portable across x86-64 platforms and unproven on arm64.
#
# THE POLICY. A golden value may be pinned exactly on CRAN only if it pins
# the OPTIMUM -- a property of the optimization problem, which every correct
# implementation reproduces. `expect_golden()` decides that at run time, per
# fixture, with no registry and no declaration:
#
#   stable under tightening -> exact pin, asserted everywhere including CRAN
#   not stable              -> on CRAN, INVARIANTS ONLY: shape, length and
#                              finiteness. No pointwise value and no
#                              magnitude band. The exact pin still runs in
#                              full locally and in CI (NOT_CRAN=true).
#
# "Stable under tightening" means: re-solve the same problem at a much
# tighter stopping tolerance and see whether the solution moves. See
# .golden_tighten_move() for why that is the right question -- in short,
# `epsilon` defines a BALL of acceptable iterates, every point of which
# reports "optimization finished", and platforms that round differently
# land on different points of it.
#
# WHAT THIS REPLACED, AND WHY. Earlier versions of this file keyed the
# decision off the solver's end state: a CONVERGED fit was assumed portable,
# and a CAPPED one was graded into "stable" (loose pointwise tolerance) or
# "chaotic" (solution-magnitude band) via a registry populated by a
# 1e-15-jitter probe. CRAN's arm64 legs falsified every part of that on
# 2.50-1:
#
#   * "Converged implies portable" is false. At the solvers' default
#     epsilon, a stock x86-64 build and an FMA-contracted one BOTH report
#     `converged` and disagree by 5.8% (type 1, multiclass3, cost 100) and
#     15% (type 3, binary, cost 1, bias 5).
#   * The axis is inverted for some fixtures. At epsilon=1e-6 a
#     `capped_refit` fit agreed to EIGHT significant figures across those
#     same builds -- better than the `converged` default-epsilon fit managed
#     to two.
#   * The jitter probe picked the wrong fixtures. It registered the `_tight`
#     epsilon variants; what a real arm64 build breaks is the `_default`
#     ones. A last-ulp jitter on the design matrix is not a model of a
#     reordered arithmetic path.
#   * The magnitude band was not a conservative fallback either: arm64 moved
#     `type_3_wi_w20_eps_default_bias_-1`'s magnitude by 10.7%, twice the
#     5e-2 band.
#
# So the registry, the two cross-platform tolerances, and the
# capped/converged gate are all gone, replaced by one measurement. There is
# no classification left for a maintainer to get wrong, and a fixture added
# later is handled without anyone having to remember this file exists.
#
# This is not lost coverage. These fixtures exist to catch vendor-upgrade
# regressions, and they do that under the exact pin, which runs on every
# developer machine and in CI. CRAN's build farm is not the regression
# suite; it is the portability check, and a value chosen arbitrarily from
# inside a stopping ball has no portable content to check. What remains on
# CRAN still catches the gross failures that would matter there: wrong
# solver selected, missing primal fallback, NaN/Inf output, shape changes.
#
# Nothing is silently dropped: every fixture is still pinned exactly, off
# CRAN. And a fixture that becomes portable later -- because a vendor
# upgrade tightens a default or fixes a solver -- starts being pinned on
# CRAN again automatically. There is no stale state to maintain.
#
# VERIFYING THIS. Do not trust a jitter probe for a release decision.
# Rebuild on x86-64 with FMA contraction forced, which reproduces arm64's
# arithmetic path (bit-identically on at least one fixture) in about a
# minute -- see maintenance/README.md.
# ---------------------------------------------------------------------------

# Default tolerances. Callers override per file where that file's own
# justification differs (e.g. the SVR grids pin at 1e-8).
GOLDEN_EXACT_TOL    <- 1e-6   # same-platform pin

# ---------------------------------------------------------------------------
# Solver end-state detection, from LIBLINEAR's own verbose log.
# ---------------------------------------------------------------------------

#' Classify a solver log into a portability-relevant end state.
#'
#' For a multiclass fit the log covers several binary subproblems; if any
#' subproblem was capped the overall fit is capped, which is the conservative
#' and correct reading.
solver_state <- function(log) {
  if (any(grepl("Switching to use", log, fixed = TRUE))) {
    "capped_refit"       # cap hit, then re-solved with the primal solver
  } else if (any(grepl("reaching max number of iterations", log, fixed = TRUE))) {
    "capped_truncated"   # cap hit, raw mid-iteration state returned
  } else if (any(grepl("reaching max number of Newton iterations", log, fixed = TRUE))) {
    "capped_newton"
  } else {
    "converged"
  }
}

golden_state_is_capped <- function(state) !identical(state, "converged")

#' Is the exact same-platform pin asserted in this run?
#'
#' Mirrors testthat::skip_on_cran()'s own condition. A predicate is used
#' rather than skip_on_cran() itself because expect_golden() is called from
#' inside loops: skip_on_cran() aborts the whole enclosing test_that() block,
#' which would silently drop every key after the first capped one -- turning
#' a portability guard into a coverage hole.
golden_exact_pin_enabled <- function() {
  identical(Sys.getenv("NOT_CRAN"), "true")
}

# ---------------------------------------------------------------------------
# Audit mode (off by default, zero cost when off)
#
# With LIBLINEAR_GOLDEN_AUDIT=true, every ll_golden_fit() records its
# parameters, its solver end state and its tightening move, and every
# expect_golden() tags that record with the fixture key and the portability
# verdict. maintenance/audit-golden-portability.R reads the log and reports
# which fixtures are pinned exactly on CRAN and which are not. Recording
# here -- rather than by shimming LiblineaR() from outside -- means it sees
# the fixture key, the real arguments and the caller's RNG state, and cannot
# silently miss a file, which an outside shim demonstrably can.
# ---------------------------------------------------------------------------
golden_audit_enabled <- function() {
  identical(Sys.getenv("LIBLINEAR_GOLDEN_AUDIT"), "true")
}

# The store lives in globalenv() deliberately. testthat re-sources every
# helper-*.R into a FRESH environment for each test file, so a store held in
# this file's own environment is recreated per file and the audit tool reads
# an empty one. It is created only in audit mode, so an ordinary test run
# never touches the global environment.
.golden_audit_env <- function() {
  nm <- ".liblinear_golden_audit"
  if (!exists(nm, envir = globalenv(), inherits = FALSE)) {
    e <- new.env(parent = emptyenv())
    e$log <- list()
    assign(nm, e, envir = globalenv())
  }
  get(nm, envir = globalenv(), inherits = FALSE)
}
golden_audit_log   <- function() .golden_audit_env()$log
golden_audit_reset <- function() {
  e <- .golden_audit_env()
  e$log <- list()
  invisible(NULL)
}
.golden_audit_add  <- function(rec) {
  e <- .golden_audit_env()
  e$log[[length(e$log) + 1L]] <- rec
}
.golden_audit_tag  <- function(key, portable) {
  e <- .golden_audit_env()
  n <- length(e$log)
  if (n > 0 && is.na(e$log[[n]]$key)) {
    e$log[[n]]$key <- key
    e$log[[n]]$portable <- portable
  }
}

.golden_rng_get <- function() {
  if (exists(".Random.seed", .GlobalEnv)) get(".Random.seed", .GlobalEnv) else NULL
}
.golden_rng_set <- function(st) {
  if (!is.null(st)) assign(".Random.seed", st, envir = .GlobalEnv)
}

# ---------------------------------------------------------------------------
# The tolerance-tightening probe: how this file decides what is portable.
# ---------------------------------------------------------------------------

# Stopping tolerance used by the probe. Must be tighter than any epsilon a
# golden fixture itself requests (the tightest in the suite is 1e-6).
GOLDEN_PROBE_EPSILON <- 1e-9

#' Largest elementwise move, relative to the reference solution's scale.
#'
#' Deliberately a MAX rather than a mean: a mean dilutes a single
#' badly-moved coefficient across the whole matrix, which is exactly how a
#' non-portable fixture would slip through.
.golden_move <- function(a, b) {
  a <- as.numeric(a); b <- as.numeric(b)
  if (length(a) != length(b) || !length(a)) return(Inf)
  if (anyNA(a) || anyNA(b) || !all(is.finite(c(a, b)))) return(Inf)
  sc <- mean(abs(b))
  if (sc == 0) max(abs(a - b)) else max(abs(a - b)) / sc
}

#' How far does this fit move when the stopping tolerance is tightened?
#'
#' THIS IS THE PORTABILITY TEST, and it replaces the capped/converged
#' distinction this file used to rely on. LIBLINEAR's dual solvers stop when
#' the maximum projected-gradient violation drops below `epsilon`. That is a
#' BALL, not a point: every iterate inside it reports "optimization
#' finished" and is, legitimately, converged. Two platforms whose arithmetic
#' differs in the last ulp land on different points of that ball, and at the
#' solvers' DEFAULT epsilon the ball is wide -- measured on 2.50-1, a stock
#' x86-64 build and an FMA-contracted one both reported `converged` and
#' disagreed by 5.8% (type 1) and 15% (type 3).
#'
#' The capped/converged axis does not capture this and is in fact inverted
#' for some fixtures: at epsilon=1e-6 a `capped_refit` fit agreed to eight
#' significant figures across those same two builds, while the `converged`
#' fit at default epsilon did not agree to two.
#'
#' Re-solving the same problem at GOLDEN_PROBE_EPSILON shrinks the ball to a
#' point. If the solution barely moves, the fixture pins the OPTIMUM, which
#' every correct implementation finds, and it can be pinned exactly on CRAN.
#' If it moves, the fixture pins one platform's arbitrary choice from inside
#' the ball, and no cross-platform pointwise assertion is defensible.
#'
#' Errs toward non-portable: anything unexpected returns Inf.
.golden_tighten_move <- function(args, w, seed_before) {
  if (is.null(w)) return(Inf)
  eps <- args$epsilon
  if (!is.null(eps) && is.finite(eps) && eps <= GOLDEN_PROBE_EPSILON) return(0)

  # CRITICAL: the dual solvers draw their coordinate permutation from R's
  # RNG, so the re-solve must start from the SAME RNG state. Getting this
  # wrong measures coordinate-ordering changes instead of the width of the
  # stopping ball, and inflates the result by orders of magnitude.
  .golden_rng_set(seed_before)
  targs <- args
  targs$epsilon <- GOLDEN_PROBE_EPSILON
  tight <- NULL
  ok <- tryCatch({
    utils::capture.output(
      tight <- suppressWarnings(do.call(LiblineaR, targs)))
    TRUE
  }, error = function(e) FALSE)
  if (!ok || is.null(tight$W)) return(Inf)
  .golden_move(w, tight$W)
}

#' Fit with LiblineaR and capture the solver's end state alongside the model.
#'
#' Seeding stays the caller's responsibility -- call set.seed() immediately
#' before this, exactly as when calling LiblineaR() directly, because the
#' dual solvers draw their coordinate permutation from R's RNG.
#'
#' verbose=TRUE is forced and its output captured, so the log is available
#' for classification without reaching the test output.
ll_golden_fit <- function(...) {
  args <- list(...)
  seed_before <- .golden_rng_get()
  args$verbose <- TRUE
  log <- utils::capture.output(
    model <- suppressWarnings(do.call(LiblineaR, args))
  )
  state <- solver_state(log)

  # Probe, then put the RNG stream back exactly where the main fit left it,
  # so that whether the probe ran is invisible to everything downstream.
  seed_after <- .golden_rng_get()
  tighten_move <- .golden_tighten_move(args, model$W, seed_before)
  .golden_rng_set(seed_after)

  if (golden_audit_enabled()) {
    .golden_audit_add(list(
      key = NA_character_,          # filled in by expect_golden()
      state = state,
      type = if (is.null(args$type)) 0 else args$type,
      cost = if (is.null(args$cost)) 1 else args$cost,
      bias = if (is.null(args$bias)) 1 else args$bias,
      epsilon = if (is.null(args$epsilon)) NA_real_ else args$epsilon,
      svr_eps = if (is.null(args$svr_eps)) NA_real_ else args$svr_eps,
      tighten_move = tighten_move))
  }

  list(model = model, log = log, state = state, tighten_move = tighten_move)
}

# ---------------------------------------------------------------------------
# The assertion
# ---------------------------------------------------------------------------

#' Is this fit's solution a portable thing to pin on CRAN?
#'
#' TWO independent things can make a golden value non-portable, and a
#' fixture must clear BOTH to be pinned exactly on the build farm. Checking
#' only one was how 2.50-1 reached CRAN broken, and then how the first
#' attempt at this policy still failed under an FMA build.
#'
#'   1. The returned iterate must come from a solve that CONVERGED. A fit
#'      stopped at its iteration cap returns a raw mid-iteration state, and
#'      -- critically -- the tightening probe below is BLIND to it: asking
#'      for a tighter tolerance just hits the same cap after the same
#'      iterations and returns the same iterate, so the probe reports no
#'      movement for a value that is pure arithmetic-path artefact.
#'      `capped_refit` is fine: the cap was hit, then the primal solver
#'      re-solved from scratch, and the returned W is that converged result
#'      (measured: agrees to 8 significant figures across a stock and an
#'      FMA-contracted build, where the raw truncated iterates do not).
#'   2. The solution must be STABLE UNDER TIGHTENING -- `epsilon` defines a
#'      ball of acceptable iterates, not a point. See
#'      .golden_tighten_move().
#'
#' Axis 1 catches what axis 2 cannot see, and vice versa: at default epsilon
#' a fit converges (axis 1 clear) but sits in a wide ball (axis 2 catches
#' it), while a capped type-3 fit at epsilon=1e-6 is in a tight ball and
#' re-caps identically (axis 2 blind, axis 1 catches it). Both were observed
#' on real builds; neither axis alone is sufficient.
golden_is_portable <- function(fit, exact_tol = GOLDEN_EXACT_TOL) {
  state <- fit$state
  move  <- fit$tighten_move
  if (is.null(state) || is.null(move)) return(FALSE)
  if (!state %in% c("converged", "capped_refit")) return(FALSE)
  is.finite(move) && move <= exact_tol
}

#' Assert a golden value under the portability policy.
#'
#' @param current   freshly computed value
#' @param target    frozen golden value
#' @param key       fixture key, used as the assertion label
#' @param fit       the ll_golden_fit() result that produced `current`.
#'                  Its $state and $tighten_move carry the portability
#'                  verdict; see golden_is_portable().
#' @param exact_tol same-platform pin tolerance, and the bar the tightening
#'                  move must clear for the pin to be asserted on CRAN too
expect_golden <- function(current, target, key, fit,
                          exact_tol = GOLDEN_EXACT_TOL) {
  portable <- golden_is_portable(fit, exact_tol)

  # In audit mode, label the fit that this assertion belongs to. Each
  # expect_golden() follows its own ll_golden_fit(), so the pending record
  # is the last one; this is what gives the audit report fixture KEYS rather
  # than parameter tuples a maintainer then has to map back by hand.
  if (golden_audit_enabled()) .golden_audit_tag(key, portable)

  if (portable) {
    # The fixture pins the optimum itself: a property of the problem, which
    # every correct implementation reproduces. Pin it everywhere.
    testthat::expect_equal(current, target, tolerance = exact_tol, label = key)
    return(invisible(NULL))
  }

  # The fixture pins one point from inside a stopping ball wider than the
  # pin tolerance. Which point is a property of the arithmetic path, so on
  # CRAN assert only what every correct implementation must produce
  # regardless of its arithmetic. Deliberately NOT asserted here: any
  # pointwise value, and the solution's magnitude -- on 2.50-1 arm64 moved
  # one fixture's magnitude by 10.7%, twice the band an earlier version of
  # this file treated as the conservative fallback.
  testthat::expect_equal(dim(current), dim(target),
                         label = paste(key, "- shape"))
  testthat::expect_equal(length(current), length(target),
                         label = paste(key, "- length"))
  testthat::expect_true(all(is.finite(as.numeric(current))),
                        label = paste(key, "- all finite"))

  # The exact pin is never dropped -- only moved off the build farm. It runs
  # in full locally and in CI (NOT_CRAN=true), where comparing against a
  # value generated on this same platform is meaningful.
  if (golden_exact_pin_enabled()) {
    testthat::expect_equal(current, target, tolerance = exact_tol,
                           label = paste(key, "- exact"))
  }
}

#' Assert a golden value whose magnitude is numerical noise.
#'
#' Some fixtures are near-degenerate by construction: the solution is zero
#' to within rounding, and the stored golden value is the rounding. A
#' RELATIVE comparison of noise against noise -- which is what
#' testthat's default tolerance performs -- asserts nothing real and fails
#' on any platform whose arithmetic path differs, which is how
#' test-vendor-golden-oneclass-svm.R's nu=0.1 key broke CRAN's arm64 legs
#' on 2.50-1. Assert instead the only portable fact: the solution is zero.
#' The exact pin still runs off CRAN.
#'
#' @param zero_tol absolute bound the whole solution must respect
expect_golden_near_zero <- function(current, target, key,
                                    zero_tol  = 1e-6,
                                    exact_tol = GOLDEN_EXACT_TOL) {
  stopifnot(max(abs(as.numeric(target))) < zero_tol)

  testthat::expect_equal(dim(current), dim(target),
                         label = paste(key, "- shape"))
  testthat::expect_true(all(is.finite(as.numeric(current))),
                        label = paste(key, "- all finite"))
  testthat::expect_lt(max(abs(as.numeric(current))), zero_tol)

  if (golden_exact_pin_enabled()) {
    testthat::expect_equal(current, target, tolerance = exact_tol,
                           label = paste(key, "- exact"))
  }
}
