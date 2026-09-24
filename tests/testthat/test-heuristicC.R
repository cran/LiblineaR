# heuristicC(): closed-form check against a hand-computed value, and
# agreement across every accepted input class (dense, vector, all 6 sparse
# classes).

test_that("heuristicC matches the closed-form formula on dense data", {
  set.seed(1)
  x <- matrix(rnorm(20), ncol = 4)
  expect_equal(heuristicC(x), 1 / mean(sqrt(rowSums(x^2))), tolerance = 1e-10)
})

test_that("heuristicC agrees across dense, vector, and all 6 sparse classes", {
  skip_if_not_installed("SparseM")
  skip_if_not_installed("Matrix")

  set.seed(1)
  x <- matrix(rnorm(40), ncol = 4)
  c_dense <- heuristicC(x)

  sparse_variants <- list(
    matrix.csr = SparseM::as.matrix.csr(x),
    matrix.csc = SparseM::as.matrix.csc(x),
    matrix.coo = SparseM::as.matrix.coo(x),
    dgCMatrix  = as(x, "dgCMatrix"),
    dgRMatrix  = as(as(x, "CsparseMatrix"), "RsparseMatrix"),
    dgTMatrix  = as(x, "TsparseMatrix")
  )
  for (cls in names(sparse_variants)) {
    expect_equal(heuristicC(sparse_variants[[cls]]), c_dense, tolerance = 1e-10, label = cls)
  }

  expect_equal(heuristicC(x[, 1]), heuristicC(x[, 1, drop = FALSE]), tolerance = 1e-10)
})
