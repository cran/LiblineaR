# Generator for tests/testthat/fixtures/vendor-golden-dual-cap-stress.rds
#
# Plain R script -- NOT a test file, never auto-run by testthat. Run it
# deliberately, from the package root, with:
#   Rscript tests/testthat/fixtures/generate-vendor-golden-dual-cap-stress.R
#
# WHAT THIS CAPTURES
# -------------------
# Vendor-upgrade Stage (iii) (v2.40 -> v2.50) risk analysis (see
# VENDOR-UPGRADE-PLAN.md) found that upstream v2.42 cut every dual
# coordinate-descent solver's iteration cap from 1000 to 300, and, on
# hitting that cap:
#   - types 1 (L2R_L2LOSS_SVC_DUAL), 7 (L2R_LR_DUAL), 12
#     (L2R_L2LOSS_SVR_DUAL) now SILENTLY SWITCH to the primal Newton
#     solver (types 2/0/11's algorithm respectively) with a tightened
#     tolerance -- a genuinely different algorithm, not just a different
#     stopping point;
#   - types 3 (L2R_L1LOSS_SVC_DUAL), 13 (L2R_L1LOSS_SVR_DUAL) have no
#     primal fallback available for their loss -- they just warn and
#     return the (possibly less-converged) 300-iteration result.
#
# Before this stage, this exact code path (the cap actually being
# reached) had never been deliberately exercised anywhere in this
# battery -- every existing dual-type fixture converges comfortably
# under 300 iterations even at very tight epsilon (checked directly:
# cost up to 1e6, epsilon down to 1e-10 on the existing near-separable/
# near-deterministic stress fixtures never got close). This fixture
# exists specifically to reach it on purpose, for all 5 affected types.
#
# DATASET
# -------
# dual_stress: n=500, p=50, built from 5 latent Gaussian factors plus
# small independent noise per feature (highly collinear columns -- the
# classic slow-coordinate-descent scenario, since CD converges slowly
# when features are strongly correlated). Both a classification target
# (thresholded at the median of a noisy linear combination) and a
# regression target (the same linear combination plus small noise) are
# derived from the same feature matrix, so all 5 types are stressed on
# structurally comparable data.
#
# PARAMETERS
# ----------
# cost = 50, epsilon = 1e-6 (classification), svr_eps = 0.01
# (regression). Chosen empirically (see VENDOR-UPGRADE-PLAN.md Stage
# (iii) section) as the smallest/fastest scenario found that reliably:
#   (a) drives every one of the 5 dual types to exactly the 300-
#       iteration cap on this dataset, AND
#   (b) for the 3 switching types, lets the primal Newton fallback
#       converge cleanly afterward (no secondary "reaching max number
#       of Newton iterations" warning) -- so the golden W values below
#       are a genuine converged optimum, not an arbitrarily-truncated
#       one, keeping this fixture fast and non-flaky.
# Larger/harder variants of this same dataset shape (e.g. n=3000, p=200)
# were tried first and DO trigger the switch, but the Newton fallback
# itself then fails to converge within its own 1000-iteration cap --
# deliberately avoided here since an unconverged golden value is a bad
# regression anchor (small legitimate solver changes could move it a
# lot) and makes the test slow for no added coverage of the risk this
# fixture targets (reaching the DUAL cap, not the primal one).
#
# WHY THIS EXISTS / WHEN TO RE-RUN
# ---------------------------------
# Stage (iii)-specific addition (v2.1 -> v2.50 program; see
# VENDOR-UPGRADE-PLAN.md). Regenerate deliberately, never silently, only
# when a reviewed change is meant to alter this exact code path (e.g. a
# future upstream release changes the 300 cap, the switch targets, or
# the underlying dual/primal solver algorithms again).

library(LiblineaR)

pkg_root <- normalizePath(".")
if (!file.exists(file.path(pkg_root, "DESCRIPTION"))) {
  stop("Run this script with the package root as the working directory.")
}

DATA_SEED <- 2024
TRAINING_SEED <- 777
N <- 500
P <- 50
K <- 5
COST <- 50
EPS <- 1e-6
SVR_EPS <- 0.01

make_dual_stress_fixture <- function() {
  set.seed(DATA_SEED)
  base <- matrix(rnorm(N * K), ncol = K)
  noise <- matrix(rnorm(N * P, sd = 0.01), ncol = P)
  loadings <- matrix(rnorm(K * P), nrow = K)
  x <- base %*% loadings + noise
  colnames(x) <- paste0("x", seq_len(P))
  w_true <- rnorm(P)
  lin <- as.vector(x %*% w_true)
  y_clf <- factor(ifelse(lin + rnorm(N, sd = 2) > stats::median(lin), "pos", "neg"))
  y_reg <- lin + rnorm(N, sd = 0.1)
  list(x = x, y_clf = y_clf, y_reg = y_reg)
}

fx <- make_dual_stress_fixture()

# type -> (target, extra args, expected behavior label -- informational
# only, the test file re-derives and asserts this from the verbose
# output text directly rather than trusting this comment).
scenarios <- list(
  type1  = list(type = 1,  y = fx$y_clf, extra = list()),
  type7  = list(type = 7,  y = fx$y_clf, extra = list()),
  type3  = list(type = 3,  y = fx$y_clf, extra = list()),
  type12 = list(type = 12, y = fx$y_reg, extra = list(svr_eps = SVR_EPS)),
  type13 = list(type = 13, y = fx$y_reg, extra = list(svr_eps = SVR_EPS))
)

results <- list()
verbose_logs <- list()
for (key in names(scenarios)) {
  sc <- scenarios[[key]]
  set.seed(TRAINING_SEED)
  args <- c(list(data = fx$x, target = sc$y, type = sc$type,
                 cost = COST, epsilon = EPS, verbose = TRUE), sc$extra)
  out <- capture.output(m <- do.call(LiblineaR, args))
  results[[key]] <- unname(m$W)
  verbose_logs[[key]] <- out
  n_dual_iter <- regmatches(out, regexpr("(?<=#iter = )[0-9]+", out, perl = TRUE))
  switched <- any(grepl("Switching", out))
  cat(sprintf("%s: dual #iter=%s, switched=%s\n", key,
              if (length(n_dual_iter)) n_dual_iter[1] else NA, switched))
}

golden <- list(
  dataset = fx,
  params = list(cost = COST, epsilon = EPS, svr_eps = SVR_EPS,
                data_seed = DATA_SEED, training_seed = TRAINING_SEED),
  results = results,
  verbose_logs = verbose_logs
)

out_path <- file.path(pkg_root, "tests", "testthat", "fixtures", "vendor-golden-dual-cap-stress.rds")
saveRDS(golden, out_path, version = 2)
cat("Wrote", length(results), "golden W matrices + verbose logs to", out_path, "\n")
