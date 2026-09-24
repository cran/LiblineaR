# LiblineaR()'s own 'data' argument accepts a plain vector for single-feature
# input, the same way predict()'s 'newx' does.

test_that("LiblineaR() accepts a plain vector data for single-feature models", {
  x1 <- c(-5, -4, -3, -2, 2, 3, 4, 5)
  y1 <- factor(rep(c("neg", "pos"), each = 4))

  m_vector <- LiblineaR(x1, y1, type = 2)
  m_matrix <- LiblineaR(matrix(x1, ncol = 1), y1, type = 2)
  expect_equal(m_vector$W, m_matrix$W)

  p <- predict(m_vector, c(-4.5, 4.5))
  expect_equal(as.character(p$predictions), c("neg", "pos"))
})
