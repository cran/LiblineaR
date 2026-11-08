### Documentation ####
#' Linear predictive models estimation based on the LIBLINEAR C/C++ Library.
#' 
#' @description
#' \code{LiblineaR} allows the estimation of predictive linear models for 
#' classification and regression, such as L1- or L2-regularized logistic 
#' regression, L1- or L2-regularized L2-loss support vector classification, 
#' L2-regularized L1-loss support vector classification and multi-class support 
#' vector classification. It also supports L2-regularized support vector regression 
#' (with L1- or L2-loss). The estimation of the models is particularly fast as 
#' compared to other libraries. The implementation is based on the LIBLINEAR C/C++ 
#' library for machine learning.
#' 
#' @details
#' For details for the implementation of LIBLINEAR, see the README file of the
#' original c/c++ LIBLINEAR library at
#' \url{https://www.csie.ntu.edu.tw/~cjlin/liblinear/}.
#' 
#' @param data 	a nxp data matrix. Each row stands for an example (sample,
#'   point) and each column stands for a dimension (feature, variable). A
#'   vector will be transformed to a n x 1 matrix. Sparse
#'   matrices of class matrix.csr, matrix.csc and matrix.coo from package
#'   SparseM are accepted. Sparse matrices of class dgCMatrix, dgRMatrix or
#'   dgTMatrix from package Matrix are also accepted. Note that C code at the
#'   core of LiblineaR package corresponds to a row-based sparse format. Hence,
#'   dgCMatrix, dgTMatrix, matrix.csc and matrix.csr inputs are first
#'   transformed into matrix.csr or dgRMatrix formats, which requires small
#'   extra computation time.
#' @param target a response vector for prediction tasks with one value for 
#'   each of the n rows of \code{data}. For classification, the values 
#'   correspond to class labels and can be a 1xn matrix, a simple vector or a 
#'   factor. For regression, the values correspond to the values to predict, and
#'   can be a 1xn matrix or a simple vector.
#' @param type \code{LiblineaR} can produce 10 types of (generalized) linear 
#'   models, by combining several types of loss functions and regularization 
#'   schemes. The regularization can be L1 or L2, and the losses can be the
#'   regular L2-loss for SVM (hinge loss), L1-loss for SVM, or the logistic loss
#'   for logistic regression. The default value for \code{type} is 0. See
#'   details below. Valid options are:
#'   \describe{
#'     \item{for multi-class classification}{
#'       \itemize{
#'         \item 0 -- L2-regularized logistic regression (primal)
#'         \item 1 -- L2-regularized L2-loss support vector classification (dual)
#'         \item 2 -- L2-regularized L2-loss support vector classification (primal)
#'         \item 3 -- L2-regularized L1-loss support vector classification (dual)
#'         \item 4 -- support vector classification by Crammer and Singer
#'         \item 5 -- L1-regularized L2-loss support vector classification
#'         \item 6 -- L1-regularized logistic regression
#'         \item 7 -- L2-regularized logistic regression (dual)
#'       }
#'     }
#'     \item{for regression}{
#'       \itemize{
#'         \item 11 -- L2-regularized L2-loss support vector regression (primal)
#'         \item 12 -- L2-regularized L2-loss support vector regression (dual)
#'         \item 13 -- L2-regularized L1-loss support vector regression (dual)
#'       }
#'     }
#'     \item{for outlier/novelty detection}{
#'       \itemize{
#'         \item 21 -- one-class support vector machine (dual). Unsupervised:
#'           \code{target} is still required (its length must match the number
#'           of rows of \code{data}) but its values are ignored -- only
#'           \code{data} is used. Requires \code{bias <= 0} (no bias term is
#'           supported for this solver). See \code{nu} below. \code{predict}
#'           returns \code{1} for inliers and \code{-1} for outliers (see
#'           \code{\link{predict.LiblineaR}}); \code{decisionValues=TRUE}
#'           additionally returns the raw signed distance to the decision
#'           boundary.
#'       }
#'     }
#'   }
#' 
#' @param cost cost of constraints violation (default: 1). Rules the trade-off
#'   between regularization and correct classification on \code{data}. It can be
#'   seen as the inverse of a regularization constant. See information on the
#'   'C' constant in details below. A usually good baseline heuristics to tune
#'   this constant is provided by the \code{heuristicC} function of this
#'   package.
#' @param epsilon set tolerance of termination criterion for optimization.
#'   If \code{NULL}, the LIBLINEAR defaults are used, which are:
#'   \describe{
#'     \item{if \code{type} is 0, 2, 5 or 6}{\code{epsilon}=0.01}
#'     \item{if \code{type} is 1, 3, 4, 7, 12 or 13}{\code{epsilon}=0.1}
#'   }
#'   
#'   % Watch out, below: in fact, \eqn{LaTeX}{ascii} is counter-intuitive.
#'   % the man pages in terminal mode are extracted from LaTeX argument, not the ascii argument !
#'   
#'   The meaning of \code{epsilon} is as follows:
#'   \describe{
#' 	    \item{if \code{type} is 0 or 2:}{
#' 			      \eqn{|f'(w)|_{2} \le \code{epsilon} \times \min (pos,neg) / l \times |f'(w_{0})|_{2}}{|f'(w)|_2 \le  epsilon * min(pos,neg) / l *|f'(w0)|_2},
#' 			      where f is the primal function and pos/neg are # of positive/negative data (default 0.01)}
#' 	    \item{if \code{type} is 11:}{
#' 			      \eqn{|f'(w)|_{2} \le \code{epsilon} \times |f'(w_{0})|_{2},}{|f'(w)|_2 \le epsilon * |f'(w0)|_2,}
#' 			      where f is the primal function (default 0.001)}
#'      \item{if \code{type} is 1, 3, 4 or 7:}{
#'            Dual maximal violation \eqn{\le \code{epsilon}}{\le epsilon} 
#'            (default 0.1)}
#'      \item{if \code{type} is 5 or 6:}{
#' 			      \eqn{|f'(w)|_\infty \le \code{epsilon}\times \min(pos,neg)/l\ |f'(w_{0})|_\infty,}{|f'(w)|_inf \le epsilon * min(pos,neg) / l*|f'(w0)|_inf,}
#' 			      where f is the primal function (default 0.01)}
#'      \item{if \code{type} is 12 or 13:}{
#' 			      \eqn{|f'(\alpha)|_1 \le \code{epsilon}\times |f'(\alpha_{0})|_1,}{|f'(alpha)|_1 \le epsilon * |f'(alpha0)|_1,}
#' 			      where f is the dual function (default 0.1)}
#' 	}
#' @param svr_eps set tolerance margin (epsilon) in regression loss function of SVR. Not used for classification methods.
#' @param bias if bias > 0, instance \code{data} becomes [\code{data}; \code{bias}]; if <= 0, no bias term added (default 1).
#' @param wi a named vector of weights for the different classes, used for
#'   asymmetric class sizes. Not all factor levels have to be supplied (default
#'   weight: 1). All components have to be named according to the corresponding
#'   class label. Not used in regression mode.
#' @param cross if an integer value k>0 is specified, a k-fold cross validation
#'   on \code{data} is performed to assess the quality of the model via a
#'   measure of the accuracy. Note that this metric might not be appropriate if
#'   classes are largely unbalanced. Default is 0.
#' @param verbose if \code{TRUE}, information are printed. Default is
#'   \code{FALSE}.
#' @param findC if \code{findC} is \code{TRUE} runs a cross-validation of \code{cross} folds to find the best cost (C) value (works only for type 0, 2 and 11).
#'        Cross validation is conducted many times under parameters C = start_C, 2*start_C, 4*start_C, 8*start_C, ..., and finds the best one with the highest cross validation accuracy
#'        (or, for type 11, the lowest cross-validation mean squared error, additionally searching \code{svr_eps} the same way).
#'        The procedure stops when the models of all folds become stable or C reaches the maximal value of 1024.
#' @param useInitC if \code{useInitC} is \code{TRUE} (default) \code{cost} is used as the smallest start_C value of the search range (\code{findC} has to be \code{TRUE}).
#'        If \code{useInitC} is \code{FALSE}, then the procedure calculates a small enough start_C itself.
#'        For \code{type}=11, \code{useInitC} additionally governs the \code{svr_eps} search: when \code{TRUE}, \code{svr_eps} sets an upper
#'        anchor point for where the (always-descending, always ending at 0) \code{svr_eps} sweep begins -- LIBLINEAR's own
#'        \code{find_parameters()} always tests every value from that anchor down to 0 and keeps whichever minimizes cross-validation error,
#'        so a small \code{svr_eps} relative to the scale of \code{target} can collapse the sweep to testing only \code{svr_eps=0}.
#'        When \code{FALSE}, the full default 20-step sweep (anchored at the largest \code{abs(target)} value) is used instead.
#' @param regularizeBias if \code{FALSE}, the bias term is excluded from the
#'   L1/L2 regularization penalty (upstream LIBLINEAR's \code{-R} flag).
#'   Default is \code{TRUE} (the bias term is regularized like every other
#'   coefficient, matching this package's behavior prior to this argument's
#'   introduction). Only valid combined with \code{bias == 1} exactly, and
#'   only for \code{type} 0, 2, 5, 6 or 11; using \code{FALSE} elsewhere, or
#'   with a different \code{bias}, raises an error. Upstream LIBLINEAR's own
#'   documentation advises against using this unless you specifically know
#'   why you need it.
#' @param nu the \code{nu} parameter of one-class SVM (\code{type} 21 only,
#'   ignored otherwise): an upper bound on the fraction of training examples
#'   that may be treated as outliers, and a lower bound on the fraction of
#'   examples that end up as support vectors. Must be in (0, 1). Default is
#'   0.5, matching upstream LIBLINEAR's own default.
#' @param ... for backwards compatibility, parameter \code{labels} may be
#'   provided instead of \code{target}. A warning will then be issued, or an
#'   error if both are present. Other extra parameters are ignored.
#' 
#' @return
#' 	If \code{cross}>0 (and \code{findC} is \code{FALSE}), the average accuracy (classification) or mean square error (regression) computed over \code{cross} runs of cross-validation is returned.\cr\cr
#'  If \code{findC} is \code{TRUE}, the best \code{cost} value found is returned; for \code{type}=11, a named list \code{list(C=..., svr_eps=...)} with both best values is returned instead of a plain number.\cr\cr
#' Otherwise, an object of class \code{"LiblineaR"} containing the fitted model is returned, including:
#' \item{TypeDetail}{A string describing the type of model fitted, as determined by \code{type}.}
#' \item{Type}{An integer corresponding to \code{type}.}
#' \item{W}{A matrix with the model weights. If \code{bias} >0, \code{W} contains p+1 columns, the last being the bias term. The columns are named according to the names of \code{data}, if provided, or \code{"Wx"} where \code{"x"} ranges from 1 to the number of dimensions. The bias term is named \code{"Bias"}.If the number of classes is 2, or if in regression mode rather than classification, the matrix only has one row. If the number of classes is k>2 (classification), it has k rows. Each row i corresponds then to a linear model discriminating between class i and all the other classes. If there are more than 2 classes, rows are named according to the class i which is opposed to the other classes.}
#' \item{Bias}{The value of \code{bias}}
#' \item{ClassNames}{A vector containing the class names. This entry is not returned in case of regression models.}
#' 
#' @references
#' 	\itemize{
#' \item 
#' For more information on LIBLINEAR itself, refer to:\cr
#' R.-E. Fan, K.-W. Chang, C.-J. Hsieh, X.-R. Wang, and C.-J. Lin.\cr
#' \emph{LIBLINEAR: A Library for Large Linear Classification,}\cr
#' Journal of Machine Learning Research 9(2008), 1871-1874.\cr
#' \url{https://www.csie.ntu.edu.tw/~cjlin/liblinear/}
#' }
#' 
#' @author Thibault Helleputte \email{thibault.helleputte@@dnalytics.com}.\cr
#'   Co-authored through version 2.10-26 by Jerome Paul and Pierre Gramme.\cr
#'   Based on C/C++-code by Chih-Chung Chang and Chih-Jen Lin
#' 
#' @note Classification models usually perform better if each dimension of the data is first centered and scaled.
#' 
#' @seealso \code{\link{predict.LiblineaR}}, \code{\link{heuristicC}}
#' 
#' @examples
#' data(iris)
#' attach(iris)
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
#' # Find the best model with the best cost parameter via 10-fold cross-validations
#' tryTypes=c(1:6)
#' tryCosts=c(1000,0.001)
#' bestCost=NA
#' bestAcc=0
#' bestType=NA
#' 
#' for(ty in tryTypes){
#' 	for(co in tryCosts){
#' 		acc=LiblineaR(data=s,target=yTrain,type=ty,cost=co,bias=1,cross=5,verbose=FALSE)
#' 		cat("Results for C=",co," : ",acc," accuracy.\n",sep="")
#' 		if(acc>bestAcc){
#' 			bestCost=co
#' 			bestAcc=acc
#' 			bestType=ty
#' 		}
#' 	}
#' }
#' 
#' cat("Best model type is:",bestType,"\n")
#' cat("Best cost is:",bestCost,"\n")
#' cat("Best accuracy is:",bestAcc,"\n")
#' 
#' # Re-train best model with best cost value.
#' m=LiblineaR(data=s,target=yTrain,type=bestType,cost=bestCost,bias=1,verbose=FALSE)
#' 
#' # Scale the test data
#' s2=scale(xTest,attr(s,"scaled:center"),attr(s,"scaled:scale"))
#' 
#' # Make prediction
#' pr=FALSE
#' if(bestType==0 || bestType==7) pr=TRUE
#' 
#' p=predict(m,s2,proba=pr,decisionValues=TRUE)
#' 
#' # Display confusion matrix
#' res=table(p$predictions,yTest)
#' print(res)
#' 
#' # Compute Balanced Classification Rate
#' BCR=mean(c(res[1,1]/sum(res[,1]),res[2,2]/sum(res[,2]),res[3,3]/sum(res[,3])))
#' print(BCR)
#' 
#' #' #############################################
#' 
#' # Example of the use of a sparse matrix of class matrix.csr :
#' 
#' if(require(SparseM)){
#' 
#'  # Sparsifying the iris dataset:
#'  iS=apply(iris[,1:4],2,function(a){a[a<quantile(a,probs=c(0.25))]=0;return(a)})
#'  irisSparse<-as.matrix.csr(iS)
#' 
#'  # Applying a similar methodology as above:
#'  xTrain=irisSparse[train,]
#'  xTest=irisSparse[-train,]
#' 
#'  # Re-train best model with best cost value.
#'  m=LiblineaR(data=xTrain,target=yTrain,type=bestType,cost=bestCost,bias=1,verbose=FALSE)
#' 
#'  # Make prediction
#'  p=predict(m,xTest,proba=pr,decisionValues=TRUE)
#' 
#' }
#' 
#' #' #############################################
#' 
#' # Example of the use of a sparse matrix of class dgCMatrix :
#' 
#' if(require(Matrix)){
#' 
#'  # Sparsifying the iris dataset:
#'  iS=apply(iris[,1:4],2,function(a){a[a<quantile(a,probs=c(0.25))]=0;return(a)})
#'  irisSparse<-as(iS,"sparseMatrix")
#' 
#'  # Applying a similar methodology as above:
#'  xTrain=irisSparse[train,]
#'  xTest=irisSparse[-train,]
#' 
#'  # Re-train best model with best cost value.
#'  m=LiblineaR(data=xTrain,target=yTrain,type=bestType,cost=bestCost,bias=1,verbose=FALSE)
#' 
#'  # Make prediction
#'  p=predict(m,xTest,proba=pr,decisionValues=TRUE)
#' 
#' }
#' 
#' #############################################
#' 
#' # Try regression instead, to predict sepal length on the basis of sepal width and petal width:
#' 
#' xTrain=iris[c(1:25,51:75,101:125),2:3]
#' yTrain=iris[c(1:25,51:75,101:125),1]
#' xTest=iris[c(26:50,76:100,126:150),2:3]
#' yTest=iris[c(26:50,76:100,126:150),1]
#' 
#' # Center and scale data
#' s=scale(xTrain,center=TRUE,scale=TRUE)
#' 
#' # Estimate MSE in cross-vaidation on a train set
#' MSECross=LiblineaR(data = s, target = yTrain, type = 13, cross = 5, svr_eps=.01)
#' 
#' # Build the model
#' m=LiblineaR(data = s, target = yTrain, type = 13, cross=0, svr_eps=.01)
#' 
#' # Test it, after test data scaling:
#' s2=scale(xTest,attr(s,"scaled:center"),attr(s,"scaled:scale"))
#' pred=predict(m,s2)$predictions
#' MSETest=mean((yTest-pred)^2)
#' 
#' # Was MSE well estimated?
#' print(MSETest-MSECross)
#' 
#' # Distribution of errors
#' print(summary(yTest-pred))
#' 
#' 
#' 
#' @keywords classif regression multivariate models optimize classes
#' 
#' @useDynLib LiblineaR trainLinear
#' 
#' @importFrom methods as 
#' 
#' @export

### Implementation ####
LiblineaR<-function(data, target, type=0, cost=1, epsilon=NULL, svr_eps=NULL, bias=1, wi=NULL, cross=0, verbose=FALSE, findC=FALSE, useInitC=TRUE, regularizeBias=TRUE, nu=0.5, ...) {
	# <Arg preparation>
  pd = .prepareData(data, "data")
  data = pd$data; n = pd$n; p = pd$p; sparse = pd$sparse; sparse2 = pd$sparse2
  rm(pd)
  
	# Backwards compatibility after renaming 'labels' to 'target'
	cc = match.call()
	if ("labels" %in% names(cc)) {
		if("target" %in% names(cc)) {
			stop("Argument 'labels' is deprecated and was renamed to 'target'. Stopping due to ambiguous call.")
		} else {
			warning("Argument 'labels' is deprecated and was renamed to 'target'.")
			target=eval(cc$labels, parent.frame())
		}
	}
	rm(cc)

	# Validate scalar numeric arguments before they reach the compiled solver:
	# check_parameter() in src/trainLinear.c only Rprintf()s a message on an
	# invalid value and returns, so an invalid cost/bias/epsilon/svr_eps must be
	# caught here to get a proper R error instead of a silently wrong model.
	if(!is.numeric(cost) || length(cost)!=1 || is.na(cost) || cost<=0)
		stop("'cost' must be a single positive number.")
	if(!(is.numeric(bias) || is.logical(bias)) || length(bias)!=1 || is.na(bias))
		stop("'bias' must be a single (non-missing) number (or TRUE/FALSE, for backward compatibility).")
	if(!is.null(epsilon) && (!is.numeric(epsilon) || length(epsilon)!=1 || is.na(epsilon)))
		stop("'epsilon' must be NULL or a single number.")
	if(!is.null(svr_eps) && (!is.numeric(svr_eps) || length(svr_eps)!=1 || is.na(svr_eps)))
		stop("'svr_eps' must be NULL or a single number.")
	if(!is.logical(regularizeBias) || length(regularizeBias)!=1 || is.na(regularizeBias))
		stop("'regularizeBias' must be TRUE or FALSE.")
	if(!is.numeric(nu) || length(nu)!=1 || is.na(nu) || nu<=0 || nu>=1)
		stop("'nu' must be a single number strictly between 0 and 1.")

	# Bias
	b = if(bias > 0) bias else -1 # to ensure backward compatibility with boolean
	
	# Type 
	typesLabels = c("L2-regularized logistic regression (primal)", "L2-regularized L2-loss support vector classification (dual)", "L2-regularized L2-loss support vector classification (primal)", "L2-regularized L1-loss support vector classification (dual)",
									"support vector classification by Crammer and Singer", "L1-regularized L2-loss support vector classification", "L1-regularized logistic regression", "L2-regularized logistic regression (dual)",
									"", "", "",
									"L2-regularized L2-loss support vector regression (primal)", "L2-regularized L2-loss support vector regression (dual)", "L2-regularized L1-loss support vector regression (dual)",
									"", "", "", "", "", "", "",
									"one-class support vector machine (dual)")
	typesLabels = gsub("[()]","",typesLabels)
	typesCodes = c("L2R_LR", "L2R_L2LOSS_SVC_DUAL", "L2R_L2LOSS_SVC", "L2R_L1LOSS_SVC_DUAL",
								 "MCSVM_CS", "L1R_L2LOSS_SVC", "L1R_LR", "L2R_LR_DUAL",
								 "", "", "",
								 "L2R_L2LOSS_SVR", "L2R_L2LOSS_SVR_DUAL", "L2R_L1LOSS_SVR_DUAL",
								 "", "", "", "", "", "", "",
								 "ONECLASS_SVM")
	types=ifelse(typesCodes=="", "", paste0(typesLabels, " (", typesCodes, ")"))
	if(!type %in% (which(types!="")-1))
		stop("Unknown model type ",type,". Expecting one of: ", paste(which(types!="")-1, collapse=", "))
	isRegression = type>=11 && type<=13
	isOneClassSVM = type==21

	# One-class SVM (type=21): LIBLINEAR's check_parameter() rejects
	# bias >= 0 outright for this solver (it has no bias-term concept),
	# rather than silently ignoring it -- checked here proactively for a
	# clear R-level error instead of the generic C-level "ERROR: ..."
	# message-and-silently-return-garbage path.
	if(isOneClassSVM && bias > 0)
		stop("'bias' must be <= 0 (no bias term) for one-class SVM (type=21); LIBLINEAR does not support a bias term for this solver.")

	# regularizeBias=FALSE (upstream's -R): check_parameter() requires
	# bias==1 exactly and restricts this to types 0, 2, 5, 6 and 11 --
	# checked here proactively for a clear R-level error, same reasoning
	# as the one-class check just above.
	if(!regularizeBias) {
		if(!isTRUE(all.equal(b, 1)))
			stop("'regularizeBias' can only be FALSE when 'bias' is exactly 1.")
		if(!(type %in% c(0,2,5,6,11)))
			stop("'regularizeBias' can only be FALSE for type 0, 2, 5, 6 or 11.")
	}

	# Epsilon: NULL or any non-positive value (including 0, which would otherwise
	# be an unsatisfiable/hanging stopping criterion) routes to LIBLINEAR's own
	# solver-appropriate default, applied in src/trainLinear.c's setup_params().
	if(is.null(epsilon) || epsilon<=0){
		epsilon = -1
	}

	# Regression margin (epsilon)
	if(is.null(svr_eps) || svr_eps<0){
		svr_eps = 0.1
		# This default value may not make sense (depends on expected precision)
		if(isRegression)
			warning("No value provided for svr_eps. Using default of 0.1")
	}
	

	# Target
	if(!is.null(dim(target))) {
		if(identical(dim(target), c(n,1L)))
			target = target[,1]
		else
			stop("Wrong dimension for target")
	}
	if(length(target)!=n){
		stop("Number of target elements disagrees with number of data instances.")
	}
	
	if(isRegression) {
		yC = as.double(target)
		isMulticlass = FALSE
		targetLabels = NULL
		nbClass = 2 # necessary for predict
		nrWi = 0
		Wi = NULL
		WiLabels = NULL
	} else if(isOneClassSVM) {
		# One-class SVM is unsupervised: solve_oneclass_svm() (src/linear.cpp)
		# never reads prob->y, and the resulting model has model_->label ==
		# NULL (set directly by train()). 'target' is still required, and
		# its length must still match nrow(data) (checked above, same as
		# every other type), but its VALUES are ignored -- forced to 0 here
		# rather than passed through, so a factor/character target can't
		# produce a confusing as.double() coercion nobody will ever look at.
		yC = rep(0, n)
		isMulticlass = FALSE
		targetLabels = NULL
		nbClass = 2 # necessary for predict; matches model_->nr_class set by train()
		nrWi = 0
		Wi = NULL
		WiLabels = NULL
	} else {
		if(is.character(target))
			target = factor(target)

		# A numeric target with a fractional part would otherwise be silently
		# truncated by as.integer() below, collapsing distinct classes together
		# (e.g. c(1.1, 1.9) both truncate to 1) with no error at all.
		if(is.numeric(target) && any(target != as.integer(target), na.rm=TRUE))
			stop("'target' contains non-integer numeric values, which is not valid for a classification 'type'. ",
					 "Use a factor (or character/integer/logical) if these are meant to be distinct class labels, ",
					 "or use a regression 'type' (11-13) if these are continuous values to predict.")

		# targetLabels are sorted by first occurrence in target ; if target is a factor, targetLabels too, with the same levels.
		targetLabels = unique(target)
		nbClass = length(targetLabels)

		yC = as.integer(target)
		
		if (nbClass<2)
			stop("Wrong number of classes ( < 2 ).")
		isMulticlass = (nbClass>2) || (nbClass==2 && type==4)
		
		# Different class penalties? Default to 1
		nrWi=nbClass
		Wi=rep(1,times=nrWi)
		names(Wi)=as.character(targetLabels)
		WiLabels=as.integer(targetLabels)
		
		if(!is.null(wi)) {
			if(is.null(names(wi)))
				stop("wi has to be a named vector!")
			
			if( !all(names(wi) %in% names(Wi)) )
				stop("Mismatch between provided names for 'wi' and class labels.")
			
			for(i in 1:length(wi)){
				Wi[names(wi)[i]]=wi[i]
			}
		}
	}
	
	# Cross-validation?
	if(cross<0){
		stop("Cross-validation argument 'cross' cannot be negative!")
	}
	else if(cross>n){
		stop("Cross-validation argument 'cross' cannot be larger than the number of samples (",n,").",sep="")
	}
	
	# Return storage preparation
	labels_ret = rep(0L, nbClass)
	nr_W = if(isMulticlass) nbClass else 1
	nc_W = if(bias > 0) p+1 else p
	W_ret=matrix(data=0, nrow=nr_W, ncol=nc_W)
	
	#
	# </Arg preparation>
	# as.double(t(X)) corresponds to rewrite X as a nxp-long vector instead of a n-rows and p-cols matrix. Rows of X are appended one at a time.
	ret <- .C("trainLinear",
			as.double(W_ret),
			as.integer(labels_ret),
			as.double(if(sparse){data@ra}else if(sparse2){data@x}else{t(data)}),
			as.double(yC),
			as.integer(n),
			as.integer(p),
			# sparse index info. 
			# If sparse or sparse2 is TRUE, then the following expression will return 1
			# And, no need for the trainLinear C function to be able to distinguish between sparse and sparse2 as we send same info.
			as.integer(sparse+sparse2),
			as.integer(if(sparse){data@ia}else if(sparse2){data@p+1}else{0}), 
			as.integer(if(sparse){data@ja}else if(sparse2){data@j+1}else{0}),

			as.double(b),
			as.integer(type),
			as.double(cost),
			as.double(epsilon),
			as.double(svr_eps),
			as.integer(nrWi),
			as.double(Wi),
			as.integer(WiLabels),
			as.integer(cross),
			as.integer(verbose),
			as.integer(findC),
			as.integer(useInitC),
			as.integer(regularizeBias),
			# bestP_ret: only meaningfully overwritten by the C side when
			# findC=TRUE and type=11 (may then come back as NA_REAL if
			# the search itself failed). Passed in as a real, finite
			# placeholder (not NA) specifically so NAOK is NOT needed
			# here -- NAOK gates the whole .C() call's incoming-value
			# check, and this package relies on that check catching
			# NA/Inf in the actual training data (see
			# test-input-validation.R); the C side is free to WRITE
			# NA_REAL back into this buffer regardless, since NAOK only
			# restricts what may be passed IN, not what a compiled
			# routine returns.
			as.double(-1),
			as.double(nu),
			PACKAGE="LiblineaR"
			)

	if(cross==0 && !findC){
		# copy_model() (src/linear.cpp) never writes labels_ret for a
		# one-class model (model_->label is NULL, set by train()), so
		# ret[[2]] would just be its uninitialized-to-0 R-side buffer --
		# treated the same as regression (no class labels) here.
		labels_ret = if(isRegression || isOneClassSVM) NULL else ret[[2]]

		# classNames is a lookup table for conversion between outputs of C code and initial levels of target
		classNames = {
			if(isRegression || isOneClassSVM) NULL
			else if(is.logical(targetLabels)) as.logical(labels_ret)
			else if(is.null(levels(targetLabels))) labels_ret
			else factor(levels(targetLabels)[labels_ret], levels=levels(targetLabels))
		}
		
		W_ret = matrix(data=ret[[1]], nrow=nr_W, ncol=nc_W)
		colnames(W_ret) = c(
			if(!is.null(colnames(data))) colnames(data) else paste0("W",1:p),
			if(bias > 0) "Bias" else c()
		)
		
		if(isMulticlass)
			rownames(W_ret) = classNames
		
		m=list()
		class(m)="LiblineaR"
		m$TypeDetail=types[type+1]
		m$Type=type
		m$W=W_ret
		m$Bias=bias
		m$ClassNames=classNames
		m$NbClass=nbClass
		return(m)
	}
	else{
		# ret[[23]] is bestP_ret: only meaningfully set (by
		# do_find_parameter_C(), src/trainLinear.c) when findC=TRUE and
		# type==11 -- NA otherwise, matching the NA_real_ this call sent in.
		if(findC && type==11)
			return(list(C=ret[[1]][1], svr_eps=ret[[23]][1]))
		return(ret[[1]][1])
	}
}

