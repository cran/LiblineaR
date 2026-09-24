#' Fast Heuristics For The Estimation Of the C Constant Of A Support Vector Machine.
#'
#' \code{heuristicC} implements a heuristics proposed by Thorsten Joachims in
#' order to make fast estimates of a convenient value for the C constant used by
#' support vector machines. This implementation only works for linear support
#' vector machines.
#'
#' @return 	A value for the C constant is returned, computed as follows:\cr
#' \eqn{\frac{1}{\frac{1}{n}\sum_{i=1}^{n}\sqrt{G[i,i]}}}{1/(1/n Sum_i=1:n sqrt(G[i,i]))}
#' where
#' \eqn{G=\code{data}\%*\%t(\code{data})}{data \%*\% t(data)}
#'
#' @param data a nxp data matrix. Each row stands for an example (sample, point)
#'   and each column stands for a dimension (feature, variable). A vector will
#'   be transformed to a n x 1 matrix. Sparse matrices of class matrix.csr,
#'   matrix.csc and matrix.coo from package SparseM are accepted, as are
#'   dgCMatrix, dgRMatrix and dgTMatrix from package Matrix -- the same classes
#'   accepted by \code{\link{LiblineaR}} and \code{\link{predict.LiblineaR}}.
#'
#' @references
#' 	\itemize{
#' \item
#' T. Joachims\cr
#' \emph{SVM light} (2002)\cr
#' \url{http://svmlight.joachims.org}
#' }
#'
#' @author Thibault Helleputte \email{thibault.helleputte@@dnalytics.com}
#'
#' @note Classification models usually perform better if each dimension of the
#'   data is first centered and scaled. If data are scaled, it is better to
#'   compute the heuristics on the scaled data as well.
#'
#' @seealso \code{\link{LiblineaR}}
#'
#' @examples
#' data(iris)
#'
#' x=iris[,1:4]
#' y=factor(iris[,5])
#' train=sample(1:dim(iris)[1],100)
#'
#' xTrain=x[train,]
#' xTest=x[-train,]
#' yTrain=y[train]
#' yTest=y[-train]
#'
#' # Center and scale data
#' s=scale(xTrain,center=TRUE,scale=TRUE)
#'
#' # Sparse Logistic Regression
#' t=6
#'
# Tune the cost parameter of a logistic regression according to the Joachim's heuristics
#' co=heuristicC(s)
#' m=LiblineaR(data=s,target=yTrain,type=t,cost=co,bias=TRUE,verbose=FALSE)
#'
#'
#' @keywords classif
#' @export

heuristicC<-function(data){
	# A vector is a single-feature input, consistent with LiblineaR()/predict().
	if(is.null(dim(data)) && !inherits(data, c("matrix.csr","matrix.csc","matrix.coo","dgCMatrix","dgRMatrix","dgTMatrix"))){
		data <- matrix(data, ncol=1)
	}

	if(inherits(data, "matrix.csr") | inherits(data, "matrix.csc") | inherits(data, "matrix.coo")){
		if(requireNamespace("SparseM", quietly=TRUE)){
			# SparseM has no rowSums method for its sparse classes, so the
			# row-wise sum of squares is computed directly from the CSR
			# representation (ra: nonzero values, ia: 1-based row pointers)
			# instead of densifying the whole matrix.
			if(inherits(data, "matrix.csc") | inherits(data, "matrix.coo"))
				data <- SparseM::as.matrix.csr(data)
			n <- data@dimension[1]
			rowSumSq <- vapply(seq_len(n), function(i){
				if(data@ia[i+1] <= data@ia[i]) return(0)
				sum(data@ra[data@ia[i]:(data@ia[i+1]-1)]^2)
			}, numeric(1))
		} else {
			stop("data inherits from 'matrix.csr', but 'SparseM' package is not available. Cannot proceed further. You could either use non-sparse matrix, install SparseM package or use sparse matrices based on Matrix package, also supported by LiblineaR.")
		}
	} else if(inherits(data, "dgCMatrix") | inherits(data, "dgRMatrix") | inherits(data, "dgTMatrix")){
		if(requireNamespace("Matrix", quietly=TRUE)){
			rowSumSq <- Matrix::rowSums(data^2)
		} else {
			stop("data inherits from 'dgCMatrix' or 'dgRMatrix', but 'Matrix' package is not available. Cannot proceed further. You could either use non-sparse matrix, install Matrix package or use sparse matrices based on SparseM package, also supported by LiblineaR.")
		}
	} else {
		rowSumSq <- rowSums(data^2)
	}

	C=1/mean(sqrt(rowSumSq))
	return(C)
}
