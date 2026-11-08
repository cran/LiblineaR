# Generator for tests/testthat/fixtures/vendor-golden-sparse-formats.rds.
#
# NOT a test file: testthat only auto-runs files matching test-*.R, and this
# is named generate-*.R specifically so it is never picked up. It is a plain
# R script, run deliberately from the package root with:
#
#   Rscript tests/testthat/fixtures/generate-vendor-golden-sparse-formats.R
#
# WHAT IT CAPTURES
# -----------------
# Golden-value regression fixture for the pending LIBLINEAR vendor upgrade
# (~v2.1 -> v2.50, touching src/linear.cpp, src/tron.cpp -> newton.cpp,
# trainLinear.c, predictLinear.c). It exists to catch a sparse-path-specific
# regression across MULTIPLE solver types -- the only other place sparse
# input is exercised, test-sparse-dense-equivalence.R, covers type 0 only
# and does not pin golden values (it just compares sparse-vs-dense trained
# fresh each run, so a regression shared by both paths would slip through
# silently).
#
# For classification types {0, 1, 3, 4, 6} (on fixture_classif()) and
# regression types {11, 12} (on fixture_regr()), for each type this script:
#   (a) trains once on the dense matrix and pins that W as the golden
#       reference for the type (so a regression in the DENSE path for these
#       types is caught here too, not only sparse-vs-dense divergence);
#   (b) trains once on each of the 6 accepted sparse-matrix classes
#       (matrix.csr/csc/coo via SparseM; dgCMatrix/dgRMatrix/dgTMatrix via
#       Matrix) and verifies (at generation time) that every one reproduces
#       the same dense W, so the pinned golden value is a genuine shared
#       reference, not an arbitrary pick;
#   (c) pins predict() output (predictions, and for the logistic types 0/6,
#       proba=TRUE probabilities) from one representative sparse class
#       (dgCMatrix) per type, to also catch a predictLinear.c sparse-path
#       regression, not just trainLinear.c's.
#
# Dual solvers among these types (1, 3, 4, 12) route randomness through R's
# own RNG (rand_between() in src/linear.cpp calls back into R), so
# set.seed(42) is issued immediately before EVERY single training call
# below (dense and each sparse class), not once at the top -- otherwise
# runs would not be reproducible relative to each other or across re-runs.
#
# PROVENANCE: captured from LiblineaR 2.10-26, vendored LIBLINEAR ~v2.1,
# already verified correct (888 pre-existing testthat assertions passing).
# This is a regression baseline, not externally-audited ground truth --
# there is no independent LIBLINEAR CLI reference to diff against for these
# exact settings. Re-run this script deliberately (never silently, never as
# part of routine test runs) to regenerate the .rds file, ONLY when a
# reviewed vendor-upgrade stage is intended to change these numeric values
# (e.g. the documented v2.40 Newton-solver rewrite), then diff the change
# against the expected magnitude before committing the updated fixture.

suppressMessages(library(LiblineaR))
suppressMessages(library(SparseM))
suppressMessages(library(Matrix))

# Locate helper-fixtures.R regardless of the current working directory --
# this script is meant to be run with cwd = package root, but source() the
# robust way in case that's not exactly true.
root <- getwd()
helper_path <- file.path(root, "tests", "testthat", "helper-fixtures.R")
if (!file.exists(helper_path)) {
  stop("Run this script with the working directory set to the package root ",
       "(expected to find tests/testthat/helper-fixtures.R under it).")
}
source(helper_path)

SEED <- 42L

sparse_variants_of <- function(x) {
  list(
    matrix.csr = SparseM::as.matrix.csr(x),
    matrix.csc = SparseM::as.matrix.csc(x),
    matrix.coo = SparseM::as.matrix.coo(x),
    dgCMatrix  = as(x, "dgCMatrix"),
    # dgRMatrix via the CsparseMatrix->RsparseMatrix route, which is not
    # deprecated (unlike as(<matrix>,"dgRMatrix") directly) -- same idiom as
    # test-sparse-dense-equivalence.R.
    dgRMatrix  = as(as(x, "CsparseMatrix"), "RsparseMatrix"),
    dgTMatrix  = as(x, "TsparseMatrix")
  )
}

# Representative sparse class used to pin predict() output, per type.
PREDICT_CLASS <- "dgCMatrix"

capture_type <- function(x, y, type, logistic, extra = list()) {
  set.seed(SEED)
  dense <- do.call(LiblineaR, c(list(data = x, target = y, type = type, cost = 1), extra))
  W_dense <- unname(dense$W)

  variants <- sparse_variants_of(x)
  max_diff <- 0
  for (cls in names(variants)) {
    set.seed(SEED)
    m <- suppressMessages(suppressWarnings(
      do.call(LiblineaR, c(list(data = variants[[cls]], target = y, type = type, cost = 1), extra))
    ))
    d <- max(abs(unname(m$W) - W_dense))
    max_diff <- max(max_diff, d)
    if (cls == PREDICT_CLASS) predict_model <- m
  }
  # Sanity check at generation time, not a testthat assertion: every sparse
  # class must reproduce the pinned dense W, or PREDICT_CLASS's predictions
  # below would not actually correspond to W_dense.
  if (max_diff > 1e-8) {
    stop(sprintf("type %d: sparse classes diverge from dense W by up to %.3g -- ",
                  type, max_diff),
         "investigate before freezing this as a golden fixture.")
  }

  xs <- variants[[PREDICT_CLASS]]
  p <- suppressMessages(suppressWarnings(
    if (logistic) predict(predict_model, xs, proba = TRUE) else predict(predict_model, xs)
  ))

  out <- list(
    W_dense = W_dense,
    predict_class = PREDICT_CLASS,
    predictions = p$predictions
  )
  if (logistic) out$probabilities <- p$probabilities
  out
}

fc <- fixture_classif()
fr <- fixture_regr()

classif_types <- c(0, 1, 3, 4, 6)
logistic_types <- c(0, 6) # proba=TRUE is only meaningful for these among our set
regr_types <- c(11, 12)

golden <- list(classif = list(), regr = list())

for (t in classif_types) {
  golden$classif[[paste0("type", t)]] <- capture_type(
    fc$x, fc$y, type = t, logistic = (t %in% logistic_types)
  )
}

for (t in regr_types) {
  golden$regr[[paste0("type", t)]] <- capture_type(
    fr$x, fr$y, type = t, logistic = FALSE, extra = list(svr_eps = 0.01)
  )
}

out_path <- file.path(root, "tests", "testthat", "fixtures", "vendor-golden-sparse-formats.rds")
saveRDS(golden, out_path, version = 2)
cat("Wrote", out_path, "\n")
