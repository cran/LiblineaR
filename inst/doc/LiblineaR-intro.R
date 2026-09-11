## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(collapse = TRUE, comment = "#>")

## ----setup--------------------------------------------------------------------
library(LiblineaR)
data(iris)

## ----types--------------------------------------------------------------------
x <- iris[, 1:4]
y <- iris[, 5]

m_lr  <- LiblineaR(x, y, type = 0)   # logistic regression
m_svm <- LiblineaR(x, y, type = 2)   # L2-loss SVM

dim(m_lr$W)   # one row per class (3 classes, multi-class problem)

## ----svr_eps------------------------------------------------------------------
xr <- as.matrix(iris[, 1:3])
yr <- iris[, 4]
m_svr <- LiblineaR(xr, yr, type = 11, svr_eps = 0.05)

## ----sparse-------------------------------------------------------------------
if (requireNamespace("Matrix", quietly = TRUE)) {
  x_sparse <- Matrix::Matrix(as.matrix(x), sparse = TRUE)
  m_sparse <- LiblineaR(x_sparse, y, type = 0)
  identical(dim(m_sparse$W), dim(m_lr$W))
}

## ----wi-----------------------------------------------------------------------
# Not all classes need to be named -- only the one(s) you want to reweight.
m_weighted <- LiblineaR(x, y, type = 0, wi = c(setosa = 5))

## ----cost---------------------------------------------------------------------
co <- heuristicC(x)
co

acc <- LiblineaR(x, y, type = 0, cost = co, cross = 5)
acc

best_cost <- LiblineaR(x, y, type = 0, findC = TRUE, cross = 5)
best_cost

m_final <- LiblineaR(x, y, type = 0, cost = best_cost)

## ----predict------------------------------------------------------------------
p <- predict(m_final, x)
mean(as.character(p$predictions) == as.character(y))

# Probabilities are only available for logistic regression (type 0, 6, 7).
p_proba <- predict(m_final, x, proba = TRUE)
head(p_proba$probabilities)

