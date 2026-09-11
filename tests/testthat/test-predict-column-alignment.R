# predict()'s documented ability to reorder/subset newx columns to match
# training column order.

test_that("predict() reorders newx columns to match training order", {
  f <- fixture_classif()
  m <- LiblineaR(f$x, f$y, type = 0)
  p_orig <- predict(m, f$x)$predictions

  newx_reordered <- f$x[, c(3, 1, 2)]
  p_reordered <- predict(m, newx_reordered)$predictions
  expect_equal(as.character(p_reordered), as.character(p_orig))
})

test_that("predict() drops extra columns not present at training time", {
  # p (the feature count sent to the C code) must match newx's column count
  # after it has been subset down to the training features by name -- the
  # transmitted data buffer has one value per row for each of those columns,
  # and a mismatched count would misalign every row after the first.
  f <- fixture_classif()
  m <- LiblineaR(f$x, f$y, type = 0)
  p_orig <- predict(m, f$x)$predictions

  newx_extra_after <- cbind(f$x, extra = seq_len(nrow(f$x)))
  expect_equal(as.character(predict(m, newx_extra_after)$predictions), as.character(p_orig))

  # Extra columns on both sides, plus reordering, all at once.
  newx_extra_both <- cbind(extra1 = 99 + seq_len(nrow(f$x)), f$x[, c(3, 1, 2)], extra2 = -seq_len(nrow(f$x)))
  expect_equal(as.character(predict(m, newx_extra_both)$predictions), as.character(p_orig))
})

test_that("predict() errors clearly when a training column is missing from newx", {
  f <- fixture_classif()
  m <- LiblineaR(f$x, f$y, type = 0)
  newx_missing <- f$x[, c(1, 2), drop = FALSE]
  expect_error(predict(m, newx_missing), "columns of 'test' and 'train' differ")
})
