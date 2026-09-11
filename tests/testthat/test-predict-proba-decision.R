# predict(..., proba=TRUE) and predict(..., decisionValues=TRUE): the
# package's headline logistic-regression differentiator.

test_that("proba=TRUE gives valid probabilities for logistic regression types (0, 6, 7)", {
  f <- fixture_classif()
  for (tt in c(0, 6, 7)) {
    m <- LiblineaR(f$x, f$y, type = tt)
    p <- predict(m, f$x, proba = TRUE)
    label <- paste("type", tt)

    expect_true(all(colnames(p$probabilities) == m$ClassNames), info = label)
    expect_equal(rowSums(p$probabilities), rep(1, nrow(f$x)), tolerance = 1e-6, label = label)
    # argmax of the probabilities must match the returned class prediction
    argmax <- colnames(p$probabilities)[apply(p$probabilities, 1, which.max)]
    expect_equal(argmax, as.character(p$predictions), label = label)
  }
})

test_that("proba=TRUE on a non-logistic type warns and is silently disabled", {
  f <- fixture_classif()
  m <- LiblineaR(f$x, f$y, type = 1)
  expect_warning(predict(m, f$x, proba = TRUE), "only supported for Logistic Regressions")
  p <- suppressWarnings(predict(m, f$x, proba = TRUE))
  expect_false("probabilities" %in% names(p))
})

test_that("decisionValues=TRUE gives a sign-consistent n x k matrix for classification", {
  f <- fixture_classif()
  m <- LiblineaR(f$x, f$y, type = 0)
  pd <- predict(m, f$x, decisionValues = TRUE)

  expect_equal(dim(pd$decisionValues), c(nrow(f$x), length(m$ClassNames)))
  expect_true(all(colnames(pd$decisionValues) == m$ClassNames))

  # Only ONE discriminant vector exists for a plain binary model
  # (src/linear.cpp predict_values(), nr_class==2, non-Crammer-Singer), so
  # only ClassNames[1]'s column ever carries a value -- ClassNames[2]'s
  # column is always exactly 0. Upstream's intended behavior for the 2-class
  # case, asserted here so a future change to that convention doesn't slip
  # by unnoticed.
  col1 <- pd$decisionValues[, m$ClassNames[1]]
  col2 <- pd$decisionValues[, m$ClassNames[2]]
  expect_true(all(col2 == 0))
  expect_true(all(col1[as.character(pd$predictions) == m$ClassNames[1]] > 0))
  expect_true(all(col1[as.character(pd$predictions) == m$ClassNames[2]] < 0))
})

test_that("decisionValues=TRUE on a regression model warns and is silently disabled", {
  f <- fixture_regr()
  m <- suppressWarnings(LiblineaR(f$x, f$y, type = 11))  # unrelated svr_eps-default notice
  expect_warning(predict(m, f$x, decisionValues = TRUE), "only supported for classification")
  p <- suppressWarnings(predict(m, f$x, decisionValues = TRUE))
  expect_false("decisionValues" %in% names(p))
})
