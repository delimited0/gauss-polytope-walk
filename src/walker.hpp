#ifndef WALKER_HPP
#define WALKER_HPP

#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

class Walker {
public: 
  Walker(const arma::vec & initial, const arma::mat & A, const arma::vec & b) :
  dim(A.n_cols), n_constr(A.n_rows), n_samples(1), initial(initial), A(A), b(b),
  curr_sample(initial)
  {}
  
  virtual ~Walker() {}
  
  virtual bool do_sample(arma::vec & x, double lazy = 0.5) {
    return false;
  }
  
  bool check_in_polytope(const arma::vec & x) {
    return arma::max((A * x - b)) < 0;
  }
  
protected:
  
  const int dim;
  const int n_constr;
  int n_samples;
  const arma::vec initial;
  const arma::mat & A;
  const arma::vec & b;
  arma::vec curr_sample;
};

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

#endif 
