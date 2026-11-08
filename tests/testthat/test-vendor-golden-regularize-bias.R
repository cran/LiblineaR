# VENDOR-UPGRADE GOLDEN BATTERY: param.regularize_bias, added in Stage (ii)
# (LIBLINEAR v2.30 -> v2.40) of the vendor upgrade of the bundled LIBLINEAR
# C/C++ source (src/linear.cpp, src/tron.cpp -> newton.cpp, trainLinear.c).
# Extended for Phase B (see VENDOR-UPGRADE-PLAN.md), which exposes this as
# LiblineaR()'s own `regularizeBias` argument (default TRUE, preserving
# every behavior pinned below) -- see the third section for that coverage.
#
# Upstream v2.40 added a `regularize_bias` field to parameter (the -R CLI
# flag: "not regularize the bias"), read by check_parameter() and by several
# solver loops in src/linear.cpp. Because `parameter` is zero-initialized and
# LiblineaR does not expose an -R option of its own, regularize_bias==0 is
# NOT a safe default: check_parameter() (src/linear.cpp, ~line 3556) rejects
# ANY bias != 1 whenever regularize_bias == 0, with
# "To not regularize bias, must specify -B 1 along with -R" -- for EVERY
# solver type, including ones that never read regularize_bias in their own
# solve loop. Stage (ii)'s fix sets param.regularize_bias = 1 unconditionally
# in setup_params() (src/trainLinear.c), which both (a) avoids the spurious
# rejection for any bias != 1, including LiblineaR's own bias<=0 "no bias
# term" encoding, and (b) restores the pre-v2.40 semantics of an
# unconditionally-regularized bias term.
#
# This file is self-referential/structural, not golden-pinned: no fixture
# file is needed or produced. It pins two behaviors directly against the
# live installed package:
#   1. bias != 1 training no longer hard-errors (the literal regression
#      Stage (ii)'s fix closes), across a solver type inside check_parameter's
#      -R allow-list (type 0) and one outside it (type 1, a dual solver).
#   2. the bias coefficient is actually L2-regularized (shrinks with cost),
#      which is the actual semantic promise of regularize_bias=1 and would
#      NOT be caught by (1) alone, nor by re-generating a golden W fixture
#      without deliberately checking for this signature.
#
# Regenerate: nothing to regenerate here (no .rds fixture) -- if a future
# vendor-upgrade stage legitimately changes this behavior (e.g. LiblineaR
# grows its own -R passthrough), update this file's expectations
# deliberately, in the same reviewed change, not silently.

# --- 1. regularize_bias is preserved (bias != 1 does not error) -----------

test_that("regularize_bias is preserved (training with bias != 1 does not error)", {
  # Risk covered: before Stage (ii)'s fix, param.regularize_bias was left at
  # its zero-initialized value (0), and check_parameter() rejects ANY bias
  # != 1 whenever regularize_bias == 0 -- "To not regularize bias, must
  # specify -B 1 along with -R" -- regardless of solver type, since the
  # check sits ahead of any type-specific branching. bias=5 (an explicit,
  # unusual bias value) and bias=-1 (LiblineaR's own internal encoding for
  # "no bias term", reached whenever the user passes bias<=0) both trigger
  # this path. Covered for type 0 (L2R_LR primal -- inside check_parameter's
  # own -R allow-list: L2R_LR/L2R_L2LOSS_SVC/L1R_L2LOSS_SVC/L1R_LR/
  # L2R_L2LOSS_SVR) and type 1 (L2R_L2LOSS_SVC_DUAL -- NOT in that allow-list,
  # confirming the pre-fix rejection was unconditional on type, not specific
  # to solvers that actually read regularize_bias in their own solve loop).
  fc <- fixture_classif()

  for (type in c(0, 1)) {
    for (bias in c(5, -1)) {
      set.seed(777)
      expect_error(
        LiblineaR(fc$x, fc$y, type = type, bias = bias),
        NA,
        label = paste("type", type, "bias", bias)
      )
    }
  }
})

# --- 2. the bias coefficient is actually L2-regularized --------------------

# A dataset with a real, non-zero (large) intercept: two well-separated
# classes offset far from the origin along both features, unlike
# fixture_classif() (helper-fixtures.R), which is symmetric around 0 (means
# -1.5/+1.5) and so fits a comparatively small bias term (~0.15-0.21 at
# cost=1, checked manually) -- not large enough to show a clean shrink
# signature. Offsetting the classes to means 10 and 20 forces a
# substantial fitted intercept.
make_offset_fixture <- function() {
  set.seed(99)
  n <- 40
  x <- rbind(
    matrix(rnorm(n / 2 * 2, mean = 10), ncol = 2),
    matrix(rnorm(n / 2 * 2, mean = 20), ncol = 2)
  )
  colnames(x) <- c("x1", "x2")
  y <- factor(rep(c("neg", "pos"), each = n / 2))
  list(x = x, y = y)
}

test_that("the bias coefficient is L2-regularized (shrinks toward 0 as cost decreases), not left unpenalized", {
  # Risk covered: this is deliberately a structural inequality test, not a
  # golden pin, so it keeps catching a regression even if someone later
  # regenerates a golden W fixture without noticing a semantic change (e.g.
  # regularize_bias silently flipped back to 0, or a future LiblineaR -R
  # passthrough defaulting to "unregularized"). If the bias term were NOT
  # L2-regularized, it would be free to fit the (large, real) intercept
  # regardless of cost, and would stay large even at a tiny cost. Because it
  # IS regularized, the penalty term (bias^2, scaled like every other
  # coefficient) dominates the objective as cost -> 0, so the fitted bias
  # must shrink toward 0 along with every other weight.
  #
  # Types 0 (L2R_LR primal) and 2 (L2R_L2LOSS_SVC primal): both Newton-
  # affected by Stage (ii)'s tron -> newton rewrite (src/newton.cpp) AND
  # inside check_parameter's -R allow-list, so both the solver-rewrite risk
  # and the regularize_bias risk are exercised on the same coefficient.
  #
  # cost=100 (large, penalty negligible) vs cost=1e-4 (tiny, penalty
  # dominates), bias=1, epsilon=1e-8 (tight, for a well-converged comparison
  # not confounded by early stopping).
  #
  # Observed ratio abs(bias_big)/abs(bias_small), checked manually before
  # picking the threshold below: ~70463x for type 0, ~1888x for type 2.
  # A factor-of-10 threshold is therefore very conservative -- comfortably
  # far below both observed ratios (by 2-3 orders of magnitude), so it will
  # not spuriously fail on ordinary numeric noise, while still being an
  # unmistakable signature that would fail hard if the bias stopped
  # shrinking with cost (e.g. ratio near 1, as an unregularized bias would
  # give).
  min_shrink_factor <- 10

  f <- make_offset_fixture()

  for (type in c(0, 2)) {
    set.seed(777)
    m_big <- LiblineaR(f$x, f$y, type = type, cost = 100, bias = 1, epsilon = 1e-8)
    set.seed(777)
    m_small <- LiblineaR(f$x, f$y, type = type, cost = 1e-4, bias = 1, epsilon = 1e-8)

    bias_big <- abs(unname(m_big$W[, "Bias"]))
    bias_small <- abs(unname(m_small$W[, "Bias"]))

    expect_true(
      bias_big > min_shrink_factor * bias_small,
      label = paste0(
        "type ", type, ": bias_big=", bias_big, " bias_small=", bias_small,
        " ratio=", bias_big / bias_small, " (expected ratio > ", min_shrink_factor, ")"
      )
    )
  }
})

# --- 3. Phase B: the R-level 'regularizeBias' argument -------------------
#
# Phase B (see VENDOR-UPGRADE-PLAN.md) exposes param.regularize_bias as
# LiblineaR()'s own 'regularizeBias' argument (default TRUE). check_parameter()
# requires bias==1 exactly and one of types {0,2,5,6,11} whenever
# regularizeBias=FALSE is requested; LiblineaR() enforces both proactively
# with a clear R-level stop() rather than letting the request reach the
# C-level "ERROR: ..." message-and-silently-return path.

test_that("regularizeBias defaults to TRUE and is validated proactively", {
  fc <- fixture_classif()

  # regularizeBias=FALSE requires bias==1 exactly.
  expect_error(
    LiblineaR(fc$x, fc$y, type = 0, bias = 5, regularizeBias = FALSE),
    "'regularizeBias' can only be FALSE when 'bias' is exactly 1",
    fixed = TRUE
  )
  expect_error(
    LiblineaR(fc$x, fc$y, type = 0, bias = -1, regularizeBias = FALSE),
    "'regularizeBias' can only be FALSE when 'bias' is exactly 1",
    fixed = TRUE
  )
  # regularizeBias=FALSE requires type in {0,2,5,6,11}.
  expect_error(
    LiblineaR(fc$x, fc$y, type = 1, bias = 1, regularizeBias = FALSE),
    "'regularizeBias' can only be FALSE for type 0, 2, 5, 6 or 11",
    fixed = TRUE
  )
  # regularizeBias is a required-shape scalar logical.
  expect_error(LiblineaR(fc$x, fc$y, regularizeBias = c(TRUE, FALSE)), "'regularizeBias'")
  expect_error(LiblineaR(fc$x, fc$y, regularizeBias = NA), "'regularizeBias'")
  expect_error(LiblineaR(fc$x, fc$y, regularizeBias = 1), "'regularizeBias'")

  # regularizeBias=TRUE (the default) is accepted for every type/bias
  # combination already covered by sections 1-2 above -- not re-asserted
  # here, just confirming the argument's default doesn't change anything.
  set.seed(777)
  m_default <- LiblineaR(fc$x, fc$y, type = 0, bias = 1)
  set.seed(777)
  m_explicit_true <- LiblineaR(fc$x, fc$y, type = 0, bias = 1, regularizeBias = TRUE)
  expect_equal(m_default$W, m_explicit_true$W)
})

test_that("regularizeBias=FALSE shrinks the bias much less with cost than regularizeBias=TRUE (structural, not golden-pinned)", {
  # Risk covered: same reasoning as section 2 above, but comparing the
  # TRUE and FALSE cases directly rather than asserting an absolute
  # "shrinks a lot" / "doesn't shrink" split -- checked manually before
  # writing this test, an unregularized bias still shrinks SOMEWHAT with
  # cost here (indirectly, through the feature weights it's fit jointly
  # with), just far less than a regularized one: observed shrink ratio
  # (bias at cost=100 / bias at cost=1e-4) was ~70463x (type 0) / ~1888x
  # (type 2) regularized, vs ~112x / ~4.4x unregularized -- a 429-629x
  # difference. A conservative 5x-smaller-ratio threshold is used below,
  # comfortably far under both observed differences.
  f <- make_offset_fixture()
  min_ratio_difference <- 5

  for (type in c(0, 2)) {
    shrink_ratio <- function(regularizeBias) {
      set.seed(777)
      m_big <- LiblineaR(f$x, f$y, type = type, cost = 100, bias = 1, epsilon = 1e-8, regularizeBias = regularizeBias)
      set.seed(777)
      m_small <- LiblineaR(f$x, f$y, type = type, cost = 1e-4, bias = 1, epsilon = 1e-8, regularizeBias = regularizeBias)
      abs(unname(m_big$W[, "Bias"])) / abs(unname(m_small$W[, "Bias"]))
    }

    ratio_regularized <- shrink_ratio(TRUE)
    ratio_unregularized <- shrink_ratio(FALSE)

    expect_true(
      ratio_regularized > min_ratio_difference * ratio_unregularized,
      label = paste0(
        "type ", type, ": ratio_regularized=", ratio_regularized,
        " ratio_unregularized=", ratio_unregularized
      )
    )
  }
})
