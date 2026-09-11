# Regression tests covering LiblineaR()'s scalar-argument validation, the
# 'wi' partial-class-weight contract, predict()'s vector-newx support, and
# the native routines' behavior on invalid input reached directly.

set.seed(1)
x2 <- rbind(
  matrix(rnorm(15 * 2, mean = -2), ncol = 2),
  matrix(rnorm(15 * 2, mean =  2), ncol = 2)
)
y2 <- factor(rep(c("neg", "pos"), each = 15))  # levels: neg=1, pos=2

test_that("'wi' may name only a subset of classes", {
  # Per the documented contract, not every class needs a weight in 'wi' --
  # unnamed classes default to weight 1.
  expect_error(m <- LiblineaR(x2, y2, wi = c(pos = 3)), NA)
  expect_s3_class(m, "LiblineaR")

  # An unknown class name in wi must still be rejected.
  expect_error(LiblineaR(x2, y2, wi = c(unknownclass = 3)), "Mismatch")
})

test_that("omitting 'epsilon' reaches the same default as epsilon=NULL", {
  expect_null(formals(LiblineaR)$epsilon)

  # type=1 is a dual coordinate-descent solver: it shuffles via R's RNG
  # (rand_between(), src/linear.cpp), so the seed must be reset immediately
  # before each call for the two runs to be directly comparable.
  set.seed(42); m_omitted <- LiblineaR(x2, y2, type = 1)
  set.seed(42); m_explicit_null <- LiblineaR(x2, y2, type = 1, epsilon = NULL)
  expect_equal(m_omitted$W, m_explicit_null$W)

  # epsilon=0 must also route to the solver default rather than an
  # effectively unsatisfiable stopping criterion.
  expect_error(LiblineaR(x2, y2, type = 1, epsilon = 0), NA)
})

test_that("predict() accepts a plain vector newx for single-feature models", {
  # The doc promises a vector newx is transformed into an n x 1 matrix.
  x1 <- matrix(c(-5, -4, -3, -2, 2, 3, 4, 5), ncol = 1)
  y1 <- factor(rep(c("neg", "pos"), each = 4))
  m1 <- LiblineaR(x1, y1, type = 2)

  p_vector <- predict(m1, c(-4.5, 4.5))
  p_matrix <- predict(m1, matrix(c(-4.5, 4.5), ncol = 1))
  expect_equal(p_vector$predictions, p_matrix$predictions)
  expect_equal(as.character(p_vector$predictions), c("neg", "pos"))
})

test_that("invalid cost/bias/epsilon/svr_eps are rejected, not silently trained", {
  expect_error(LiblineaR(x2, y2, cost = 0), "cost")
  expect_error(LiblineaR(x2, y2, cost = -1), "cost")
  expect_error(LiblineaR(x2, y2, cost = c(1, 2)), "cost")
  expect_error(LiblineaR(x2, y2, bias = c(1, 2)), "bias")
  expect_error(LiblineaR(x2, y2, bias = NA_real_), "bias")
  expect_error(LiblineaR(x2, y2, bias = NA), "bias")
  expect_error(LiblineaR(x2, y2, epsilon = c(0.1, 0.2)), "epsilon")
  expect_error(LiblineaR(x2, y2, svr_eps = c(0.1, 0.2)), "svr_eps")

  # bias=TRUE/FALSE (boolean backward-compatibility form, used by heuristicC's
  # own roxygen example) must be accepted alongside plain numeric bias values.
  expect_error(LiblineaR(x2, y2, bias = TRUE), NA)
  expect_error(LiblineaR(x2, y2, bias = FALSE), NA)
})

test_that("trainLinear() frees its buffers when check_parameter() rejects, even via direct .C() calls", {
  # LiblineaR() rejects cost<=0 at the R level, so trainLinear.c's
  # check_parameter() rejection path isn't reachable through the public API.
  # trainLinear is nonetheless a registered, directly callable native
  # routine, so this calls it exactly as LiblineaR() would, but with an
  # invalid cost, to exercise that path directly. This can only assert the
  # call completes cleanly and repeatedly without crashing -- confirming the
  # freed memory itself requires a leak checker (valgrind/ASan), which a CI
  # sanitizer job should run.
  n <- nrow(x2); p <- ncol(x2)
  run_invalid <- function() {
    .C("trainLinear",
       as.double(matrix(0, nrow = 1, ncol = p)),
       as.integer(c(0L, 0L)),
       as.double(t(x2)),
       as.double(as.integer(y2)),
       as.integer(n),
       as.integer(p),
       as.integer(0),
       as.integer(0),
       as.integer(0),
       as.double(-1),
       as.integer(0),
       as.double(0),    # cost = 0 -> rejected by check_parameter()
       as.double(-1),
       as.double(0.1),
       as.integer(2),
       as.double(c(1, 1)),
       as.integer(c(1L, 2L)),
       as.integer(0),
       as.integer(0),
       as.integer(0),
       as.integer(1),
       PACKAGE = "LiblineaR")
  }
  expect_error(invisible(capture.output(for (i in 1:20) run_invalid())), NA)
})

test_that("predictLinear() errors cleanly on an invalid solver type instead of crashing", {
  # predict.LiblineaR() already gates object$Type at the R level; predictLinear
  # is nonetheless a registered, directly callable native routine, so this
  # bypasses the R gate on purpose to exercise load_model()'s NULL-return
  # path (linear.cpp) and predictLinear()'s own guard (predictLinear.c) directly.
  m <- LiblineaR(x2, y2, type = 2)
  n <- nrow(x2); p <- ncol(x2)

  capture.output(
    expect_error(
      .C("predictLinear",
         as.double(numeric(n)),
         as.double(t(x2)),
         as.double(m$W),
         as.integer(0),
         as.double(-1),
         as.integer(0),
         as.double(-1),
         as.integer(m$NbClass),
         as.integer(p),
         as.integer(n),
         as.integer(0),
         as.integer(0),
         as.integer(0),
         as.double(m$Bias),
         as.integer(seq_len(length(m$ClassNames))),
         as.integer(999L),  # invalid solver type
         PACKAGE = "LiblineaR"),
      regexp = "[Ii]nvalid model|unknown"
    )
  )
})

test_that("a would-overflow dense allocation size errors cleanly instead of corrupting memory", {
  # setup_problem() (src/trainLinear.c) computes the dense-case allocation
  # size for x_space in a 64-bit-safe type and errors if it exceeds INT_MAX,
  # since the C struct fields it feeds are 32-bit int. LiblineaR() itself
  # cannot be driven into that regime in a fast unit test (it would require
  # actually allocating a huge matrix), so this calls the registered
  # trainLinear native routine directly with n/p values chosen so that
  # n*p+n alone exceeds INT_MAX, backed by small dummy data buffers -- the
  # guard must fire before the fill loop ever reads them.
  huge_n <- 50000L
  huge_p <- 50000L  # huge_n*huge_p + huge_n ~= 2.5e9 > .Machine$integer.max

  expect_error(
    .C("trainLinear",
       as.double(matrix(0, nrow = 1, ncol = huge_p)),
       as.integer(c(0L, 0L)),
       as.double(rep(0, huge_p)),   # dummy row; never actually read
       as.double(rep(0, huge_n)),
       as.integer(huge_n),
       as.integer(huge_p),
       as.integer(0),
       as.integer(0),
       as.integer(0),
       as.double(-1),
       as.integer(0),
       as.double(1),
       as.double(-1),
       as.double(0.1),
       as.integer(2),
       as.double(c(1, 1)),
       as.integer(c(1L, 2L)),
       as.integer(0),
       as.integer(0),
       as.integer(0),
       as.integer(1),
       PACKAGE = "LiblineaR"),
    regexp = "too large|exceeds"
  )
})
