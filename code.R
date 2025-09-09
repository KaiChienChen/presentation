# PGA / OGA / CGA implementations in R
# Author: ChatGPT (example implementation)
# Notes:
# - X: n x p numeric matrix (columns are predictors)
# - y: numeric vector (for PGA/OGA continuous), for CGA should be 0/1 or -1/+1 (we use 0/1)
# - standardize: whether to column-standardize X (center=FALSE, scale=TRUE) -- centering handled separately
# - K: maximum number of steps
# - For CGA we use logistic regression (glm binomial) as the refit step.

pga <- function(X, y, K = 50, standardize = TRUE, verbose = TRUE) {
  n <- nrow(X); p <- ncol(X)
  if (standardize) {
    # scale columns to have unit norm (but keep mean of x_j possibly not zero)
    Xs <- scale(X, center = TRUE, scale = apply(X, 2, sd))
  } else {
    Xs <- X
  }
  y <- as.numeric(y)
  y_hat <- rep(0, n)
  residual <- y - y_hat
  selected <- integer(0)
  gammas <- numeric(0)
  yhat_path <- matrix(0, nrow = n, ncol = K)
  for (k in 1:K) {
    # compute score for each candidate j not yet selected: abs(sum(residual * xj))
    remaining <- setdiff(1:p, selected)
    if (length(remaining) == 0) break
    scores <- sapply(remaining, function(j) abs(sum(residual * Xs[, j])))
    jstar <- remaining[which.max(scores)]
    # compute optimal gamma (one-dim LS)
    numer <- sum(residual * Xs[, jstar])
    denom <- sum(Xs[, jstar]^2)
    if (denom == 0) { gamma <- 0 } else { gamma <- numer / denom }
    # update
    y_hat <- y_hat + gamma * Xs[, jstar]
    residual <- y - y_hat
    selected <- c(selected, jstar)
    gammas <- c(gammas, gamma)
    yhat_path[, k] <- y_hat
    if (verbose) message(sprintf("PGA step %d: selected X%d, gamma=%.4f", k, jstar, gamma))
  }
  list(method = "PGA",
       selected = selected,
       gammas = gammas,
       y_hat = y_hat,
       residual = residual,
       yhat_path = yhat_path[, 1:length(selected), drop = FALSE])
}


oga <- function(X, y, K = 50, standardize = TRUE, verbose = TRUE) {
  n <- nrow(X); p <- ncol(X)
  if (standardize) {
    Xs <- scale(X, center = TRUE, scale = apply(X, 2, sd))
  } else {
    Xs <- X
  }
  y <- as.numeric(y)
  residual <- y
  selected <- integer(0)
  models <- list()
  y_hat <- rep(0, n)
  yhat_path <- matrix(0, nrow = n, ncol = K)
  for (k in 1:K) {
    remaining <- setdiff(1:p, selected)
    if (length(remaining) == 0) break
    scores <- sapply(remaining, function(j) abs(sum(residual * Xs[, j])))
    jstar <- remaining[which.max(scores)]
    selected <- c(selected, jstar)
    # Refit OLS on selected set (with intercept)
    Xsel <- Xs[, selected, drop = FALSE]
    df <- data.frame(y = y, Xsel)
    # build formula dynamically
    colnames(df) <- c("y", paste0("V", 1:ncol(Xsel)))
    form <- as.formula(paste("y ~ ", paste(colnames(df)[-1], collapse = " + ")))
    fit <- lm(form, data = df)
    y_hat <- predict(fit)
    residual <- y - y_hat
    models[[k]] <- fit
    yhat_path[, k] <- y_hat
    if (verbose) message(sprintf("OGA step %d: selected X%d, model df %d", k, jstar, length(selected)))
  }
  list(method = "OGA",
       selected = selected,
       models = models,
       y_hat = y_hat,
       residual = residual,
       yhat_path = yhat_path[, 1:length(models), drop = FALSE])
}


cga <- function(X, y, K = 50, standardize = TRUE, verbose = TRUE, eps = 1e-12) {
  # CGA for logistic (y in {0,1})
  n <- nrow(X); p <- ncol(X)
  if (!all(y %in% c(0,1))) stop("For CGA, y must be 0/1.")
  if (standardize) {
    Xs <- scale(X, center = TRUE, scale = apply(X, 2, sd))
  } else {
    Xs <- X
  }
  selected <- integer(0)
  models <- list()
  # initialize beta as zero and p_i = 0.5
  eta <- rep(0, n)           # linear predictor
  p_i <- 1 / (1 + exp(-eta)) # predicted probability
  for (k in 1:K) {
    # gradient for logistic log-likelihood w.r.t. beta_j is sum_i x_ij * (y_i - p_i)
    remaining <- setdiff(1:p, selected)
    if (length(remaining) == 0) break
    grads <- sapply(remaining, function(j) sum(Xs[, j] * (y - p_i)))
    jstar <- remaining[which.max(abs(grads))]
    selected <- c(selected, jstar)
    # Refit logistic on selected set
    Xsel <- Xs[, selected, drop = FALSE]
    df <- data.frame(y = y, Xsel)
    colnames(df) <- c("y", paste0("V", 1:ncol(Xsel)))
    form <- as.formula(paste("y ~ ", paste(colnames(df)[-1], collapse = " + ")))
    fit <- glm(form, family = binomial(), data = df, control = glm.control(maxit = 50))
    # update linear predictor and p_i
    eta <- predict(fit, type = "link")
    p_i <- predict(fit, type = "response")
    # store
    models[[k]] <- fit
    if (verbose) message(sprintf("CGA step %d: selected X%d, |grad|=%.4g", k, jstar, abs(grads[which.max(abs(grads))])))
    # small stopping if perfect separation or tiny changes
    if (any(is.na(p_i)) || any(p_i < eps) || any(1 - p_i < eps)) {
      if (verbose) message("CGA stopping: probabilities collapsed or NA.")
      break
    }
  }
  list(method = "CGA",
       selected = selected,
       models = models,
       last_prob = p_i)
}


# -------------------------
# Example usage (simulated data)
# -------------------------
set.seed(42)

# Example for PGA / OGA (regression)
n <- 100; p <- 10
X <- matrix(rnorm(n * p), nrow = n)
beta_true <- rep(0, p)
beta_true[c(2,5)] <- c(3, -2)
y_reg <- X %*% beta_true + rnorm(n, sd = 1)

# Run PGA
res_pga <- pga(X, y_reg, K = 10, standardize = TRUE, verbose = TRUE)

# Run OGA
res_oga <- oga(X, y_reg, K = 10, standardize = TRUE, verbose = TRUE)

# Example for CGA (classification)
# create binary outcome depending on X[,2] and X[,5]
lin <- 1.5 * X[,2] - 2 * X[,5] + rnorm(n, sd = 1)
p_true <- 1 / (1 + exp(-lin))
y_bin <- rbinom(n, size = 1, prob = p_true)

res_cga <- cga(X, y_bin, K = 10, standardize = TRUE, verbose = TRUE)

# Check selected variables
cat("PGA selected:", res_pga$selected, "\n")
cat("OGA selected:", res_oga$selected, "\n")
cat("CGA selected:", res_cga$selected, "\n")

