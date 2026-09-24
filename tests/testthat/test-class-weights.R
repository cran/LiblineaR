# Class weighting with 'wi': partial-class-naming and unknown-class-name
# validation are already covered in test-p0-fixes.R. This covers the actual
# mechanism -- reweighting shifts the decision boundary in the expected
# direction, not just "does it run".

test_that("naming only the minority class in wi shifts recall toward it", {
  set.seed(7)
  n_maj <- 90; n_min <- 10
  x <- rbind(
    matrix(rnorm(n_maj * 2, mean =  0.4), ncol = 2),
    matrix(rnorm(n_min * 2, mean = -0.4), ncol = 2)
  )
  y <- factor(c(rep("majority", n_maj), rep("minority", n_min)))

  m_unweighted <- LiblineaR(x, y, type = 0, cost = 1)
  p_unweighted <- predict(m_unweighted, x)$predictions
  recall_unweighted <- mean(p_unweighted[y == "minority"] == "minority")

  # Only the minority class is named; majority keeps its default weight (1).
  m_weighted <- LiblineaR(x, y, type = 0, cost = 1, wi = c(minority = 15))
  p_weighted <- predict(m_weighted, x)$predictions
  recall_weighted <- mean(p_weighted[y == "minority"] == "minority")

  expect_lt(recall_unweighted, 0.3)
  expect_gt(recall_weighted, 0.8)
})
