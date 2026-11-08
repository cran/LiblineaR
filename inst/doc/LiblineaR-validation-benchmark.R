## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(collapse = TRUE, comment = "#>")

## ----setup--------------------------------------------------------------------
library(LiblineaR)

## ----oneclass-----------------------------------------------------------------
x <- iris[, 1:4]
y <- iris[, 5]

m_oc <- LiblineaR(x, y, type = 21, bias = -1, nu = 0.1)
p_oc <- predict(m_oc, newx = x, decisionValues = TRUE)
table(p_oc$predictions)
head(p_oc$decisionValues)

## ----regularizebias-----------------------------------------------------------
m_default <- LiblineaR(x, y, type = 0, bias = 1)
m_unreg   <- LiblineaR(x, y, type = 0, bias = 1, regularizeBias = FALSE)

## ----findc-svr----------------------------------------------------------------
xr <- as.matrix(iris[, 1:3])
yr <- iris[, 4]

best <- LiblineaR(xr, yr, type = 11, findC = TRUE, cross = 5)
best

m_svr <- LiblineaR(xr, yr, type = 11, cost = best$C, svr_eps = best$svr_eps)

## ----fidelity-table-----------------------------------------------------------
fidelity <- data.frame(
  type = c(0L, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 11L, 12L, 13L),
  solver = c("L2R_LR (primal)", "L2R_L2LOSS_SVC_DUAL", "L2R_L2LOSS_SVC (primal)",
             "L2R_L1LOSS_SVC_DUAL", "MCSVM_CS", "L1R_L2LOSS_SVC", "L1R_LR",
             "L2R_LR_DUAL", "L2R_L2LOSS_SVR (primal)", "L2R_L2LOSS_SVR_DUAL",
             "L2R_L1LOSS_SVR_DUAL"),
  max_abs_diff_v210 = c(1.72e-10, 9.65e-03, 1.98e-10, 3.60e-02, 1.38e-02,
                         3.93e-01, 6.82e-01, 7.32e-03, 1.84e-10, 7.93e-02, 1.71e-01),
  max_abs_diff_v250 = c(1.75e-10, 9.65e-03, 1.93e-10, 3.60e-02, 1.38e-02,
                         3.93e-01, 6.82e-01, 7.32e-03, 1.59e-10, 7.93e-02, 1.71e-01)
)
knitr::kable(fidelity, digits = 4,
  col.names = c("type", "solver", "max|diff| vs CLI (2.10)", "max|diff| vs CLI (2.50)"))

## ----upgrade-table------------------------------------------------------------
upgrade <- data.frame(
  type = c(0L, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 11L, 12L, 13L),
  solver = fidelity$solver,
  max_abs_diff = c(0.0789, 0, 0.2281, 0, 0, 0, 0, 0, 0.0021, 0, 0)
)
knitr::kable(upgrade, digits = 4, col.names = c("type", "solver", "max|diff| (2.10-26 vs. 2.50-1)"))

## ----large-table--------------------------------------------------------------
large <- data.frame(
  type = c(0L, 1L, 7L, 11L, 12L),
  solver = c("L2R_LR (primal)", "L2R_L2LOSS_SVC_DUAL", "L2R_LR_DUAL",
             "L2R_L2LOSS_SVR (primal)", "L2R_L2LOSS_SVR_DUAL"),
  metric_old = c(0.9188, 0.9219, 0.9200, 1.0234, 1.0502),
  metric_new = c(0.9181, 0.9188, 0.9200, 1.0235, 1.0234),
  time_old_s = c(0.133, 0.632, 1.035, 0.075, 3.400),
  time_new_s = c(0.147, 0.288, 0.998, 0.086, 1.201),
  speedup = c(0.90, 2.19, 1.04, 0.87, 2.83)
)
knitr::kable(large, digits = 4,
  col.names = c("type", "solver", "accuracy/MSE (2.10-26)", "accuracy/MSE (2.50-1)",
                "time 2.10-26 (s)", "time 2.50-1 (s)", "speedup"))

