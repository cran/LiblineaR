# CORE: each of the 6 accepted sparse-matrix classes must train and predict
# identically to the equivalent dense matrix. This directly targets the
# duplicated sparse-dispatch code in R/LiblineaR.R and R/predict.R -- a bug
# specific to one class's index conversion would otherwise go undetected
# indefinitely, since each class is otherwise only ever exercised on its own.
#
# All 6 classes reproduce the dense coefficients and predictions exactly on
# this fixture (max|W_sparse - W_dense| == 0), so the tolerance below is
# generous, not tuned to just barely pass.

test_that("all 6 accepted sparse-matrix classes give the same model and predictions as dense input", {
  skip_if_not_installed("SparseM")
  skip_if_not_installed("Matrix")

  f <- fixture_classif()
  dense <- LiblineaR(f$x, f$y, type = 0)
  dense_predictions <- predict(dense, f$x)$predictions

  csr <- SparseM::as.matrix.csr(f$x)
  sparse_variants <- list(
    matrix.csr = csr,
    matrix.csc = SparseM::as.matrix.csc(f$x),
    matrix.coo = SparseM::as.matrix.coo(f$x),
    dgCMatrix  = as(f$x, "dgCMatrix"),
    # dgRMatrix via the CsparseMatrix->RsparseMatrix route, which is not
    # deprecated (unlike as(<matrix>,"dgRMatrix") directly, which LiblineaR's
    # own internal dispatch code still uses and triggers a Matrix warning for).
    dgRMatrix  = as(as(f$x, "CsparseMatrix"), "RsparseMatrix"),
    dgTMatrix  = as(f$x, "TsparseMatrix")
  )

  for (cls in names(sparse_variants)) {
    xs <- sparse_variants[[cls]]
    m <- suppressMessages(suppressWarnings(LiblineaR(xs, f$y, type = 0)))
    expect_equal(unname(m$W), unname(dense$W), tolerance = 1e-6, label = paste(cls, "W"))

    p_sparse <- suppressMessages(suppressWarnings(predict(m, xs)))$predictions
    expect_equal(as.character(p_sparse), as.character(dense_predictions), label = paste(cls, "predictions"))
  }
})
