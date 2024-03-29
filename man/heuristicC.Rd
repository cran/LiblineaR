% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/heuristicC.R
\name{heuristicC}
\alias{heuristicC}
\title{Fast Heuristics For The Estimation Of the C Constant Of A Support Vector Machine.}
\usage{
heuristicC(data)
}
\arguments{
\item{data}{a nxp data matrix. Each row stands for an example (sample, point)
and each column stands for a dimension (feature, variable)}
}
\value{
A value for the C constant is returned, computed as follows:\cr
\eqn{\frac{1}{\frac{1}{n}\sum_{i=1}^{n}\sqrt{G[i,i]}}}{1/(1/n Sum_i=1:n sqrt(G[i,i]))}
where
\eqn{G=\code{data}\%*\%t(\code{data})}{data \%*\% t(data)}
}
\description{
\code{heuristicC} implements a heuristics proposed by Thorsten Joachims in
order to make fast estimates of a convenient value for the C constant used by
support vector machines. This implementation only works for linear support
vector machines.
}
\note{
Classification models usually perform better if each dimension of the
  data is first centered and scaled. If data are scaled, it is better to
  compute the heuristics on the scaled data as well.
}
\examples{
data(iris)

x=iris[,1:4]
y=factor(iris[,5])
train=sample(1:dim(iris)[1],100)

xTrain=x[train,]
xTest=x[-train,]
yTrain=y[train]
yTest=y[-train]

# Center and scale data
s=scale(xTrain,center=TRUE,scale=TRUE)

# Sparse Logistic Regression
t=6

co=heuristicC(s)
m=LiblineaR(data=s,labels=yTrain,type=t,cost=co,bias=TRUE,verbose=FALSE)


}
\references{
\itemize{
\item 
T. Joachims\cr
\emph{SVM light} (2002)\cr
\url{http://svmlight.joachims.org}
}
}
\seealso{
\code{\link{LiblineaR}}
}
\author{
Thibault Helleputte \email{thibault.helleputte@dnalytics.com}
}
\keyword{classif}
