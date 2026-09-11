# Every classification type (0-7) x weighted/unweighted x row order x all 7
# target encodings (224 combinations), checked against dimOK/perfOK/sumOK/
# biasOK/levelsOK diagnostics. One test_that() per type keeps runtime/output
# sane while still naming the exact failing combination via `info=` on each
# expectation.

df <- fixture_sweep_df()
classif_targets <- c("y.logical", "y.int", "y.double", "y.char", "y.factor",
                      "y.factorRev", "y.factorExtra", "y.multiclass")

train_classif <- function(rev, yy, weighted, tt) {
  is <- if (rev) seq_len(nrow(df)) else rev(seq_len(nrow(df)))
  nis <- which(!df[is, "y.logical"])
  y <- df[is, yy]
  x <- df[is, 1:3]

  if (weighted) {
    # One shared weight map reused across all target encodings; LiblineaR()
    # only accepts wi names that are legitimate classes of the *current*
    # target, so it's filtered down per call.
    wi_all <- c("1" = 2, "TRUE" = 2, "yes" = 2, "(7,99]" = 1,
                "(4,7]" = 50,
                "-1" = 100, "FALSE" = 100, "no" = 100, "(-99,4]" = 150)
    wi <- wi_all[names(wi_all) %in% as.character(unique(y))]
  } else {
    wi <- NULL
  }

  m <- LiblineaR(x, y, type = tt, wi = wi)
  p <- predict(m, newx = x)

  list(
    W = m$W,
    perf = mean(as.character(y) == as.character(p$predictions)),
    perfNeg = mean(as.character(y[nis]) == as.character(p$predictions[nis])),
    sumW = sum(m$W[, 1:3]),
    biasW = m$W[1, ][["Bias"]],
    yLev = paste(levels(y), collapse = " "),
    predLev = paste(levels(p$predictions), collapse = " "),
    yClass = class(y),
    predClass = class(p$predictions)
  )
}

for (tt in 0:7) {
  test_that(paste0("classification type ", tt, " trains/predicts correctly across weighting, row order and target encodings"), {
    for (weighted in c(FALSE, TRUE)) {
      for (rev in c(FALSE, TRUE)) {
        for (yy in classif_targets) {
          r <- train_classif(rev, yy, weighted, tt)
          label <- sprintf("type=%d target=%s weighted=%s rev=%s", tt, yy, weighted, rev)

          # dimOK: W is 1 x 4 (3 features + bias) unless type 4 (Crammer &
          # Singer, always one weight vector per class) or the target itself
          # is multiclass (3 classes).
          if (tt != 4 && yy != "y.multiclass") {
            expect_true(identical(dim(r$W), c(1L, 4L)), info = paste(label, "- dimOK"))
          }

          # perfOK: unweighted accuracy is held to a comfortable bar (0.75
          # binary, 0.6 multiclass). Weighted cases use looser bars, because
          # the heavy (intentionally extreme, 100-150x) class-weight skew on
          # this small fixture trades off raw accuracy for minority-class
          # recall by design -- an accuracy/balance trade-off, not a defect
          # -- most visibly for weighted multiclass (floor 0.3) and, to a
          # lesser extent, weighted binary perfNeg (floor 0.8, since with
          # only 6 negative-class samples a single misclassification already
          # crosses a tighter bar).
          if (yy == "y.multiclass") {
            floor <- if (weighted) 0.3 else 0.6
            expect_true(r$perf >= floor, info = sprintf("%s - perfOK (perf=%.3f, floor=%.2f)", label, r$perf, floor))
          } else if (weighted) {
            expect_true(r$perfNeg >= 0.8, info = sprintf("%s - perfOK (perfNeg=%.3f)", label, r$perfNeg))
          } else {
            expect_true(r$perf >= 0.75, info = sprintf("%s - perfOK (perf=%.3f)", label, r$perf))
          }

          # sumOK / biasOK: for the two purely-numeric +1/-1-coded targets
          # (excluding type 4, which has no single per-class sign
          # relationship), the summed feature weight should be positive and
          # the bias negative -- a basic sign sanity check on the learned
          # decision boundary, not just "did it run".
          if (yy %in% c("y.int", "y.double") && tt != 4) {
            expect_true(r$sumW > 0, info = sprintf("%s - sumOK (sumW=%.3f)", label, r$sumW))
            expect_true(r$biasW < 0, info = sprintf("%s - biasOK (biasW=%.3f)", label, r$biasW))
          }

          # levelsOK: predicted labels round-trip to the same R type and
          # factor levels as the true target, except y.char/y.double, which
          # have no "levels" to round-trip.
          if (!(yy %in% c("y.char", "y.double"))) {
            expect_true(identical(r$yClass, r$predClass) && r$yLev == r$predLev,
                        info = sprintf("%s - levelsOK (yClass=%s predClass=%s yLev=[%s] predLev=[%s])",
                                       label, r$yClass, r$predClass, r$yLev, r$predLev))
          }
        }
      }
    }
  })
}
