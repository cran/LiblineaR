# Every regression type (11-13) x row order (6 combinations), checked
# against dimOK/perfOK diagnostics -- same pattern as
# test-train-types-classification.R.

df <- fixture_sweep_df()

train_regr <- function(rev, tt) {
  is <- if (rev) seq_len(nrow(df)) else rev(seq_len(nrow(df)))
  y <- df[is, "y.regr"]
  x <- df[is, 1:3]

  m <- LiblineaR(x, y, type = tt, svr_eps = 0.1)
  p <- predict(m, newx = x)

  list(
    W = m$W,
    perf = RSquared(p$predictions, y)
  )
}

for (tt in 11:13) {
  test_that(paste0("regression type ", tt, " trains/predicts correctly across row order"), {
    for (rev in c(FALSE, TRUE)) {
      r <- train_regr(rev, tt)
      label <- sprintf("type=%d rev=%s", tt, rev)

      # dimOK: W is 1 x 4 (3 features + bias) -- regression models are
      # never multiclass-shaped.
      expect_true(identical(dim(r$W), c(1L, 4L)), info = paste(label, "- dimOK"))

      # perfOK: R-squared on the training data should be reasonably high
      # for this cleanly-linear-with-noise fixture.
      expect_true(r$perf >= 0.75, info = sprintf("%s - perfOK (R2=%.3f)", label, r$perf))
    }
  })
}
