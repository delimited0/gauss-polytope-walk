get_polytope_constraints <- function(lb, ub, mu, L) {
  
  d <- length(lb)
  
  inf_idx <- c(is.infinite(lb), is.infinite(ub))
  
  A <- rbind(-diag(d), diag(d))[!inf_idx, ]
  A <- A %*% L
  
  b <- c(mu - lb, -mu + ub)[!inf_idx]
  
  return(list(A = A, b = b))
}

default_initial <- function(A, b) {
  
}

#' @export
rtmvn <- function(n, mu, Sigma, lb, ub, method, r, initial = NULL) {
  
  constraints <- get_polytope_constraints(lb, ub, mu, t(chol(Sigma)))
  rtmvn2(n, mu, Sigma, constraints$A, constraints$b, method, r, initial)
}

#' @export
rtmvn2 <- function(n, mu, Sigma, A, b, method, r, initial = NULL, L = NULL) {
  
  if (is.null(L)) {
    L <- t(chol(Sigma))
  }
  
  if (method == "vaidya") {
    std_samples <- t(generate_vaidya_samples(n, initial, A, b, r))
  }
  
  samples <- std_samples %*% t(L) + matrix(rep(mu, n), nrow = n, byrow = TRUE)
}