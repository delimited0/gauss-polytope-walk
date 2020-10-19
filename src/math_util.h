#ifndef MATH_UTIL_H
#define MATH_UTIL_H

#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

double gaussian_density(const arma::vec & x, const arma::vec & mu, 
                        const arma::mat & sqrt_cov);

double gaussian_density_ratio(const arma::vec & x, const arma::vec & y);


#endif 