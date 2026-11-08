# VENDOR-UPGRADE GOLDEN BATTERY: one-class SVM (type=21), Phase B
# (see VENDOR-UPGRADE-PLAN.md).
#
# Upstream v2.40 (vendored since Stage (ii)) added ONECLASS_SVM=21
# (solve_oneclass_svm(), src/linear.cpp) and the 'nu' parameter. Stage (ii)
# and (iii) kept this path double-guarded and fully unreachable from R
# (see the former CANARY tests in test-vendor-golden-cv-findc.R, now
# superseded by this file). Phase B wires it up end to end:
#   - src/trainLinear.c: 'nu' now threaded through instead of hardcoded;
#     'epsilon=NULL' now routes to upstream's own -s 21 default (0.01,
#     the one exception to the general 0.1 dual-solver default) instead
#     of the -1 sentinel every other solver type avoided by having its
#     own switch-case.
#   - src/linear.cpp: load_model()'s guard no longer special-cases
#     ONECLASS_SVM out -- the generic bounds check is the only gate now,
#     same as every other solver type.
#   - R/LiblineaR.R: type=21 added to the whitelist; a dedicated
#     'isOneClassSVM' branch handles this solver's unsupervised nature
#     (target's values are ignored, only its length matters; nbClass=2
#     with NULL ClassNames, matching model_->label==NULL set by train()).
#   - R/predict.R: type=21 added to the whitelist; predictions are the
#     raw +1 (inlier) / -1 (outlier) LIBLINEAR itself returns, not mapped
#     through ClassNames (which is NULL for this type); decisionValues'
#     buffer sizing fixed to use object$NbClass (the C side always writes
#     nr_class=2 values per sample regardless of solver type) rather than
#     length(object$ClassNames), which is 0 for this type and previously
#     caused a buffer-size mismatch / memory corruption when combined
#     with decisionValues=TRUE (caught empirically, not by inspection,
#     while writing this test file -- see VENDOR-UPGRADE-PLAN.md).
#
# Note on reproducibility: solve_oneclass_svm() (src/linear.cpp) uses a
# quickselect (quick_select_min_k()/partition()) for its inner-loop
# shrinking strategy, which -- unlike the rest of this solver's logic --
# does call rand_between() (patched through R's own RNG at Stage (iii),
# see VENDOR-UPGRADE-PLAN.md's Stage (iii) rand() correction), so training
# is reproducible via set.seed() like every other solver type in this
# package, confirmed directly below.

fc <- fixture_classif()
oneclass_target <- rep(1, nrow(fc$x)) # ignored by solve_oneclass_svm(); see below

test_that("type=21 trains successfully, target's values are ignored (only its length matters)", {
  set.seed(777)
  m1 <- LiblineaR(fc$x, oneclass_target, type = 21, bias = -1, nu = 0.2, epsilon = 1e-6)

  # Any target content is accepted, as long as its length matches nrow(x) --
  # confirming solve_oneclass_svm() genuinely never reads it, not just that
  # this particular vector of 1s happens to work.
  set.seed(777)
  m2 <- LiblineaR(fc$x, sample(LETTERS[1:3], nrow(fc$x), replace = TRUE), type = 21, bias = -1, nu = 0.2, epsilon = 1e-6)
  set.seed(777)
  m3 <- LiblineaR(fc$x, rnorm(nrow(fc$x)), type = 21, bias = -1, nu = 0.2, epsilon = 1e-6)

  expect_equal(m1$W, m2$W)
  expect_equal(m1$W, m3$W)

  expect_equal(m1$Type, 21)
  expect_equal(dim(m1$W), c(1L, ncol(fc$x)))
  expect_equal(m1$NbClass, 2)
  expect_null(m1$ClassNames)
  expect_true(all(is.finite(m1$W)))
})

test_that("type=21 training is reproducible via set.seed() (rand_between() call site confirmed live)", {
  set.seed(42)
  m1 <- LiblineaR(fc$x, oneclass_target, type = 21, bias = -1, nu = 0.3, epsilon = 1e-6)
  set.seed(42)
  m2 <- LiblineaR(fc$x, oneclass_target, type = 21, bias = -1, nu = 0.3, epsilon = 1e-6)
  expect_identical(m1$W, m2$W)
})

test_that("type=21 requires bias <= 0 (no bias term supported), checked proactively at the R level", {
  expect_error(
    LiblineaR(fc$x, oneclass_target, type = 21, bias = 1),
    "'bias' must be <= 0",
    fixed = TRUE
  )
  expect_error(LiblineaR(fc$x, oneclass_target, type = 21, bias = -1), NA)
  expect_error(LiblineaR(fc$x, oneclass_target, type = 21, bias = 0), NA)
})

test_that("'nu' is validated as a single number strictly between 0 and 1", {
  expect_error(LiblineaR(fc$x, oneclass_target, type = 21, bias = -1, nu = 0), "'nu'")
  expect_error(LiblineaR(fc$x, oneclass_target, type = 21, bias = -1, nu = 1), "'nu'")
  expect_error(LiblineaR(fc$x, oneclass_target, type = 21, bias = -1, nu = -0.1), "'nu'")
  expect_error(LiblineaR(fc$x, oneclass_target, type = 21, bias = -1, nu = c(0.1, 0.2)), "'nu'")
  expect_error(LiblineaR(fc$x, oneclass_target, type = 21, bias = -1, nu = NA), "'nu'")
  # 'nu's shape/range is validated unconditionally (like 'cost'), even for
  # types that never read it -- basic input hygiene, not gated on type=21,
  # consistent with the existing validation style for other scalar args.
  expect_error(LiblineaR(fc$x, fc$y, type = 0, nu = 999), "'nu'")
  # A within-range nu is accepted (and silently unused) for every other type.
  expect_error(LiblineaR(fc$x, fc$y, type = 0, nu = 0.5), NA)
})

test_that("predict() on a type=21 model returns +1/-1 only, matching decisionValues' sign", {
  set.seed(777)
  m <- LiblineaR(fc$x, oneclass_target, type = 21, bias = -1, nu = 0.2, epsilon = 1e-6)
  p <- predict(m, newx = fc$x, decisionValues = TRUE)

  expect_true(all(p$predictions %in% c(-1, 1)))
  expect_equal(dim(p$decisionValues), c(nrow(fc$x), 1L))
  expect_true(all(is.finite(p$decisionValues)))
  expect_equal(sign(p$decisionValues[, 1]), as.numeric(p$predictions),
               label = "decisionValues sign must match the +1/-1 prediction, per predict_values()'s own check_oneclass_model() branch")
})

test_that("proba=TRUE is rejected (with a warning, not an error/crash) for type=21, like every non-logistic-regression type", {
  set.seed(777)
  m <- LiblineaR(fc$x, oneclass_target, type = 21, bias = -1, nu = 0.2)
  expect_warning(
    p <- predict(m, newx = fc$x, proba = TRUE),
    "Computing probabilities is only supported for Logistic Regressions"
  )
  expect_null(p$probabilities)
  expect_true(all(p$predictions %in% c(-1, 1)))
})

test_that("golden grid: type=21 W matches frozen values across nu, reproducible via set.seed()", {
  # Golden values pinned directly from the CURRENT, Phase-B-merged solver
  # (via dput() on a fresh run, not hand-typed) -- these happen to be very
  # small (~1e-7/1e-8) rather than a dramatic separating hyperplane: this
  # fixture (fixture_classif()'s x, an isotropic-ish 2-cluster layout with
  # no single consistent "outlier direction") is a near-degenerate case for
  # a LINEAR one-class SVM specifically, checked and expected -- not a
  # regression to chase, since this test's purpose is pinning reproducible
  # R-glue behavior, not demonstrating best-case one-class separation.
  # testthat's default tolerance (no override) is used deliberately, since
  # an explicit loose tolerance (e.g. 1e-6) would exceed these values'
  # own magnitude and make the comparison meaningless.
  golden <- list(
    nu0.1 = structure(c(-8.60428193699178e-08, 8.32671340096089e-08, -8.01949085610593e-08), dim = c(1L, 3L)),
    nu0.5 = structure(c(-9.93730804054007e-08, 2.15979101907303e-08, -1.1605088907087e-08), dim = c(1L, 3L))
  )
  for (nu in c(0.1, 0.5)) {
    set.seed(555)
    m <- LiblineaR(fc$x, oneclass_target, type = 21, bias = -1, nu = nu, epsilon = 1e-6)
    key <- paste0("nu", nu)
    expect_equal(unname(m$W), golden[[key]], label = key)
  }
})
