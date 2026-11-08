# VENDOR-UPGRADE GOLDEN GRID: class weight (wi) x epsilon x bias.
#
# Safety net for the upcoming vendor upgrade of the bundled LIBLINEAR C/C++
# source (src/linear.cpp, src/tron.cpp -> newton.cpp, trainLinear.c,
# predictLinear.c) from ~v2.1 to v2.50. Pins exact W from the CURRENT,
# verified-correct solver across type x wi x epsilon x bias, on a fixed
# imbalanced 90/10 2-class dataset -- exactly the parameters implicated in
# the 2.10-25 epsilon-default bug and its parsnip reverse-dependency break.
#
# Fixture provenance: tests/testthat/fixtures/vendor-golden-classweight-
# epsilon-bias.rds is produced by tests/testthat/fixtures/generate-vendor-
# golden-classweight-epsilon-bias.R (see that script's header for the full
# provenance comment, the "regenerate deliberately" caveat, and why this
# fixture is a regression baseline rather than externally-audited ground
# truth). Do not regenerate it as a side effect of a failing test -- only
# after a reviewed vendor-upgrade stage is meant to change these values.
#
# This file re-derives the same fixture inline (own copy, seed 2026) rather
# than depending on the generator script's internal helper, so the test
# keeps working even if the generator script is edited later; the generator
# and this file are kept in sync by hand.

make_fixture <- function() {
  set.seed(2026)
  n_maj <- 90; n_min <- 10
  x <- rbind(
    matrix(rnorm(n_maj * 2, mean =  0.4), ncol = 2),
    matrix(rnorm(n_min * 2, mean = -0.4), ncol = 2)
  )
  colnames(x) <- c("x1", "x2")
  y <- factor(c(rep("majority", n_maj), rep("minority", n_min)))
  list(x = x, y = y)
}

types <- c(0, 1, 2, 3)
wi_list <- list(
  unweighted = NULL,
  w5  = c(minority = 5),
  w20 = c(minority = 20)
)
epsilon_list <- list(
  default = NULL,
  e0.1    = 0.1,
  e0.01   = 0.01,
  e1em6   = 1e-6
)
bias_list <- c(-1, 1)

default_epsilon_for_type <- function(type) {
  if (type %in% c(0, 2)) 0.01 else 0.1
}

# Observed magnitude of frozen W entries spans ~1e-15 (near-zero terms) to
# ~1.87 (see generator script output), so an absolute tolerance in the
# 1e-6-1e-8 range comfortably separates float noise from a real solver-
# behaviour change without being so loose it would mask one. Primal types
# (0, 2) converge to machine-precision-tight optima under TRON, so those get
# a tighter tolerance than dual coordinate-descent types (1, 3), whose exact
# iterate path is theoretically RNG-order-dependent (rand_between) even
# though it is deterministic given a fixed seed.
tol_primal <- 1e-8
tol_dual   <- 1e-6

test_that("type 0 (L2R_LR primal) W matches frozen golden grid across wi x epsilon x bias", {
  # Risk: any change to the TRON-based logistic-regression primal solver
  # (src/tron.cpp -> newton.cpp) or to how epsilon/wi/bias are threaded into
  # it from trainLinear.c would move these values.
  ref <- readRDS(test_path("fixtures", "vendor-golden-classweight-epsilon-bias.rds"))
  f <- make_fixture()
  type <- 0
  for (wi_name in names(wi_list)) {
    for (eps_name in names(epsilon_list)) {
      for (bias in bias_list) {
        key <- paste("type", type, "wi", wi_name, "eps", eps_name, "bias", bias, sep = "_")
        set.seed(4242)
        m <- LiblineaR(f$x, f$y, type = type, cost = 1,
                        epsilon = epsilon_list[[eps_name]], bias = bias,
                        wi = wi_list[[wi_name]])
        expect_equal(unname(m$W), ref[[key]], tolerance = tol_primal, label = key)
      }
    }
  }
})

test_that("type 1 (L2R_L2LOSS_SVC_DUAL) W matches frozen golden grid across wi x epsilon x bias", {
  # Risk: any change to the dual coordinate-descent loop (solve_l2r_l1l2_svc
  # in src/linear.cpp) or to its use of R's RNG via rand_between() would
  # move these values, including changes that alter iterate order without
  # changing the final tolerance test.
  ref <- readRDS(test_path("fixtures", "vendor-golden-classweight-epsilon-bias.rds"))
  f <- make_fixture()
  type <- 1
  for (wi_name in names(wi_list)) {
    for (eps_name in names(epsilon_list)) {
      for (bias in bias_list) {
        key <- paste("type", type, "wi", wi_name, "eps", eps_name, "bias", bias, sep = "_")
        set.seed(4242)
        m <- LiblineaR(f$x, f$y, type = type, cost = 1,
                        epsilon = epsilon_list[[eps_name]], bias = bias,
                        wi = wi_list[[wi_name]])
        expect_equal(unname(m$W), ref[[key]], tolerance = tol_dual, label = key)
      }
    }
  }
})

test_that("type 2 (L2R_L2LOSS_SVC primal) W matches frozen golden grid across wi x epsilon x bias", {
  # Risk: same TRON primal machinery as type 0 but a different loss
  # function (squared hinge vs logistic) -- catches a regression specific
  # to L2-loss SVC's objective/gradient code path in src/linear.cpp.
  ref <- readRDS(test_path("fixtures", "vendor-golden-classweight-epsilon-bias.rds"))
  f <- make_fixture()
  type <- 2
  for (wi_name in names(wi_list)) {
    for (eps_name in names(epsilon_list)) {
      for (bias in bias_list) {
        key <- paste("type", type, "wi", wi_name, "eps", eps_name, "bias", bias, sep = "_")
        set.seed(4242)
        m <- LiblineaR(f$x, f$y, type = type, cost = 1,
                        epsilon = epsilon_list[[eps_name]], bias = bias,
                        wi = wi_list[[wi_name]])
        expect_equal(unname(m$W), ref[[key]], tolerance = tol_primal, label = key)
      }
    }
  }
})

test_that("type 3 (L2R_L1LOSS_SVC_DUAL) W matches frozen golden grid across wi x epsilon x bias", {
  # Risk: dual solver with the hinge (L1) loss variant -- a different code
  # path from type 1's squared-hinge dual in solve_l2r_l1l2_svc; also
  # exercises rand_between() reproducibility for a second dual loss type.
  ref <- readRDS(test_path("fixtures", "vendor-golden-classweight-epsilon-bias.rds"))
  f <- make_fixture()
  type <- 3
  for (wi_name in names(wi_list)) {
    for (eps_name in names(epsilon_list)) {
      for (bias in bias_list) {
        key <- paste("type", type, "wi", wi_name, "eps", eps_name, "bias", bias, sep = "_")
        set.seed(4242)
        m <- LiblineaR(f$x, f$y, type = type, cost = 1,
                        epsilon = epsilon_list[[eps_name]], bias = bias,
                        wi = wi_list[[wi_name]])
        expect_equal(unname(m$W), ref[[key]], tolerance = tol_dual, label = key)
      }
    }
  }
})

test_that("epsilon = NULL and epsilon = <documented per-solver default> give IDENTICAL W", {
  # Risk: this is the exact invariant the 2.10-25 epsilon-default bug broke
  # (a wrong hardcoded default in the C glue diverging from what NULL
  # actually resolved to, silently changing convergence and breaking the
  # parsnip reverse-dependency). If a future vendor-upgrade stage changes a
  # solver's internal default epsilon, this must be updated deliberately in
  # lockstep with R/LiblineaR.R's @param epsilon documentation, not left to
  # drift silently.
  ref <- readRDS(test_path("fixtures", "vendor-golden-classweight-epsilon-bias.rds"))
  f <- make_fixture()
  for (type in types) {
    set.seed(4242)
    m_null <- LiblineaR(f$x, f$y, type = type, cost = 1, epsilon = NULL, bias = 1)
    set.seed(4242)
    m_explicit <- LiblineaR(f$x, f$y, type = type, cost = 1,
                             epsilon = default_epsilon_for_type(type), bias = 1)

    key_null <- paste0("defaultcheck_type", type, "_null")
    key_explicit <- paste0("defaultcheck_type", type, "_explicit")

    # Freeze against the golden fixture (drift vs. the verified-correct
    # solver)...
    expect_equal(unname(m_null$W), ref[[key_null]], tolerance = 1e-8,
                 label = paste0("type ", type, " epsilon=NULL vs golden"))
    expect_equal(unname(m_explicit$W), ref[[key_explicit]], tolerance = 1e-8,
                 label = paste0("type ", type, " epsilon=default vs golden"))

    # ...and directly against each other (the actual invariant under test):
    # NULL must resolve to exactly the documented default, bit-for-bit.
    expect_equal(unname(m_null$W), unname(m_explicit$W), tolerance = 0,
                 label = paste0("type ", type, " NULL == explicit default epsilon"))
  }
})
