# cross=5 CV accuracy on a sparsified iris dataset, for each of the 6
# accepted sparse-matrix classes, compared against dense. Verified directly
# first: dense and all 6 sparse classes give the identical accuracy
# (0.7333333) under the same seed, so this compares each sparse class
# against a dense reference exactly, not against a loose threshold.

test_that("cross-validated accuracy on sparsified iris matches dense, for every sparse-matrix class", {
  skip_if_not_installed("SparseM")
  skip_if_not_installed("Matrix")

  # Zero out the bottom quartile of each numeric column to sparsify iris.
  iS <- apply(iris[, 1:4], 2, function(a) { a[a < quantile(a, probs = 0.25)] <- 0; a })
  y <- iris[, 5]

  set.seed(1)
  dense_acc <- LiblineaR(data = as.matrix(iS), target = y, type = 0, cost = 0.1, bias = 1, cross = 5)

  sparse_variants <- list(
    matrix.csr = SparseM::as.matrix.csr(iS),
    matrix.csc = SparseM::as.matrix.csc(iS),
    matrix.coo = SparseM::as.matrix.coo(iS),
    dgCMatrix  = as(iS, "dgCMatrix"),
    dgRMatrix  = as(as(iS, "CsparseMatrix"), "RsparseMatrix"),
    dgTMatrix  = as(iS, "TsparseMatrix")
  )

  for (cls in names(sparse_variants)) {
    set.seed(1)
    acc <- suppressMessages(suppressWarnings(
      LiblineaR(data = sparse_variants[[cls]], target = y, type = 0, cost = 0.1, bias = 1, cross = 5)
    ))
    expect_equal(acc, dense_acc, tolerance = 1e-6, label = cls)
  }
})
