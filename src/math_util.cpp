#include "math_util.h"

double gaussian_density(const arma::vec & x, const arma::vec & mu, 
                        const arma::mat & sqrt_cov) {
  arma::vec c = sqrt_cov * (x - mu);
  
  double logdet_val;
  double sign;
  arma::log_det(logdet_val, sign, sqrt_cov);
  
  return std::exp(-0.5 * arma::dot(c, c)) * std::exp(logdet_val) * sign;
}

double gaussian_density_ratio(const arma::vec & x, const arma::vec & y) {
  return std::exp(-0.5 * ( arma::dot(y, y) - arma::dot(x, x) ));
}