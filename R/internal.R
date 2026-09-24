# Internal helpers, not exported. Kept in one place so LiblineaR() and
# predict.LiblineaR() cannot drift apart on how they interpret 'data'/'newx'.

# Coerces a plain vector to an n x 1 matrix, detects and normalizes the
# sparse matrix classes both entry points accept (row-sorting, and
# column-based/coordinate-based -> row-based conversion), and returns the
# dimensions and sparse-format flags the C interface needs.
# 'argname' is used only in the error messages, so they keep reading
# "data inherits from..." from LiblineaR() and "newx inherits from..." from
# predict.LiblineaR(), as before this was factored out.
.prepareData <- function(data, argname = "data") {
  sparse = FALSE
  sparse2 = FALSE

  if(is.null(dim(data)) && !inherits(data, c("matrix.csr","matrix.csc","matrix.coo","dgCMatrix","dgRMatrix","dgTMatrix"))){
    data <- matrix(data, ncol=1)
  }

  if(sparse <- (inherits(data, "matrix.csr") | inherits(data, "matrix.csc") | inherits(data, "matrix.coo")) ){
    if(requireNamespace("SparseM",quietly=TRUE)){
      # trying to handle the sparse matrix case with SparseM package
      if(inherits(data,"matrix.csc") | inherits(data,"matrix.coo")){
        # Transform column-based sparse matrix format and coordinate-based sparse
        # matrix format of class matrix.csc or matrix.coo into row-based sparse
        # matrix format of class matrix.csr.
        data<-SparseM::as.matrix.csr(data)
      }
      data = SparseM::t(SparseM::t(data)) # make sure column index are sorted
      n = data@dimension[1]
      p = data@dimension[2]
    } else {
      stop(argname, " inherits from 'matrix.csr', but 'SparseM' package is not available. Cannot proceed further. You could either use non-sparse matrix, install SparseM package or use sparse matrices based on Matrix package, also supported by LiblineaR.")
    }
  } else if(sparse2 <- (inherits(data, "dgCMatrix") | inherits(data,"dgRMatrix") | inherits(data, "dgTMatrix"))) {
    if(requireNamespace("Matrix",quietly=TRUE)){
      # trying to handle the sparse matrix case with Matrix package
      if(inherits(data,"dgCMatrix") | inherits(data,"dgTMatrix")){
        # Transform column-based sparse matrix format and triplets-based sparse
        # matrix format of class dgCMatrix or dgTMatrix into row-based sparse
        # matrix format of class dgRMatrix.
        data<-as(as(data,"matrix"),"dgRMatrix")
      }
      data = Matrix::t(Matrix::t(data)) # make sure column index are sorted
      n = dim(data)[1]
      p = dim(data)[2]
    } else {
      stop(argname, " inherits from 'dgCMatrix' or 'dgRMatrix', but 'Matrix' package is not available. Cannot proceed further. You could either use non-sparse matrix, install Matrix package or use sparse matrices based on SparseM package, also supported by LiblineaR.")
    }
  } else {
    # Nb samples
    n=dim(data)[1]
    # Nb features
    p=dim(data)[2]
  }

  list(data=data, n=n, p=p, sparse=sparse, sparse2=sparse2)
}
