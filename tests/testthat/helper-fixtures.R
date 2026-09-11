# Coefficient of determination (R-squared), by Pierre Gramme.
RSquared <- function(pred, true) {
  ss_tot <- sum((true - mean(true))^2)
  ss_res <- sum((true - pred)^2)
  1 - ss_res / ss_tot
}

# Shared fixtures for the CORE numeric-correctness test files
# (test-reference-coefficients.R, test-solver-equivalence.R,
# test-sparse-dense-equivalence.R). Not a test file itself -- testthat loads
# any tests/testthat/helper-*.R before running tests, without treating it as
# one.

# A small, fixed, cleanly separable 2-class dataset -- deliberately easy to
# separate so different solvers converge to comparable decision boundaries
# rather than differing mainly due to the classification problem being hard.
fixture_classif <- function() {
  set.seed(123)
  n <- 40
  x <- rbind(
    matrix(rnorm(n / 2 * 3, mean = -1.5), ncol = 3),
    matrix(rnorm(n / 2 * 3, mean =  1.5), ncol = 3)
  )
  colnames(x) <- c("x1", "x2", "x3")
  y <- factor(rep(c("neg", "pos"), each = n / 2))
  list(x = x, y = y)
}

# A small, fixed linear-with-noise regression dataset.
fixture_regr <- function() {
  set.seed(321)
  n <- 40
  x <- matrix(rnorm(n * 3), ncol = 3)
  colnames(x) <- c("x1", "x2", "x3")
  y <- as.vector(x %*% c(1.5, -2, 0.5)) + rnorm(n, sd = 0.1)
  list(x = x, y = y)
}

# N=10 samples, 3 numeric features, one regression target (y.regr) and 7
# classification target encodings of the same underlying signal (logical,
# int, double, char, factor, factor with reversed levels, factor with an
# extra unused level, and a 3-level multiclass cut). Used by
# test-train-types-classification.R and test-train-types-regression.R.
fixture_sweep_df <- function() {
  set.seed(1)
  N <- 10
  df <- data.frame(
    x1 = (1:N) / N * 10 + 2 * rnorm(N),
    x2 = (1:N) / N * 10 + 2 * rnorm(N),
    x3 = (1:N) / N * 10 + 2 * rnorm(N)
  )
  df$y.regr <- apply(as.matrix(df), 1, mean) + 2 * rnorm(N)
  df$y.logical <- df$y.regr > 5.5
  df$y.int <- ifelse(df$y.logical, 1L, -1L)
  df$y.double <- as.double(df$y.int)
  df$y.char <- as.character(df$y.int)
  df$y.factor <- factor(df$y.int)
  df$y.factorRev <- factor(df$y.int, levels = rev(levels(df$y.factor)))
  df$y.factorExtra <- factor(df$y.int, levels = c(-1, 1, 99), labels = c("no", "yes", "maybe"))
  df$y.multiclass <- cut(df$y.regr, breaks = c(-99, 4, 7, 99))
  df
}
