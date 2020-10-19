#include "vaidya_walker.hpp"
#include "math_util.h"

bool VaidyaWalker::do_sample(arma::vec & new_sample, const double lazy) {
  proposal(new_sample);
  this->n_samples++;
  
  double u = R::runif(0.0, 1.0);
  
  if (u < lazy && check_in_polytope(new_sample) && accept_reject_reverse(new_sample)) {
    this->curr_sample = new_sample;
    return true;
  }
  else {
    new_sample = this->curr_sample;
    return false;
  }
}

void VaidyaWalker::proposal(arma::vec & new_sample) {
  arma::vec gaussian_step = arma::randn(this->dim);
  
  // get hessian
  arma::mat new_sqrt_inv_hess = arma::zeros(this->dim, this->dim);
  sqrt_inv_hess_barrier(this->curr_sample, new_sqrt_inv_hess);
  
  new_sample = this->curr_sample + this->r / 
    std::sqrt(std::sqrt( (double) this->dim * this->n_constr)) *
    (new_sqrt_inv_hess * gaussian_step);
}

bool VaidyaWalker::accept_reject_reverse(const arma::vec & new_sample) {
  // get hessian on x
  arma::mat new_sqrt_inv_hess_x = arma::zeros(this->dim, this->dim);
  sqrt_inv_hess_barrier(this->curr_sample, new_sqrt_inv_hess_x);
  
  // get hessian on y
  arma::mat new_sqrt_inv_hess_y = arma::zeros(this->dim, this->dim);
  sqrt_inv_hess_barrier(new_sample, new_sqrt_inv_hess_y);
  
  double scale = this->r / std::sqrt(std::sqrt( (double) this->dim * this->n_constr));
  double p_y_to_x = gaussian_density(this->curr_sample, new_sample, 
                                     arma::inv_sympd(new_sqrt_inv_hess_y) / scale);
  double p_x_to_y = gaussian_density(new_sample, this->curr_sample, 
                                     arma::inv_sympd(new_sqrt_inv_hess_x) / scale);
  double p_stdgauss_y_to_x = gaussian_density_ratio(this->curr_sample, new_sample);
  
  double ar_ratio = std::min(1.0, p_stdgauss_y_to_x * p_y_to_x / p_x_to_y);
  
  double u = R::runif(0.0, 1.0);
  if (u > ar_ratio) {
    return false;
  }
  
  return true;
}

void VaidyaWalker::sqrt_inv_hess_barrier(const arma::vec & new_sample, 
                                         arma::mat & new_sqrt_inv_hess) {
  
  arma::vec inv_slack = 1 / (this->b - this->A * new_sample);
  
  arma::mat half_hess = arma::diagmat(inv_slack) * this->A;
  arma::mat new_hess = half_hess.t() * half_hess;
  
  arma::mat new_hess_inv = arma::inv_sympd(new_hess);
  
  // compute leverage scores
  arma::vec score = arma::sum(((half_hess * new_hess_inv) % half_hess), 1);
  
  // compute Vaidya hessian
  arma::mat vaidya_new_hess = 
    half_hess.t() * arma::diagmat(score) * half_hess + 
    ( (double) this->dim / this->n_constr) * new_hess;
  
  // compute eigenvectors and eigenvalues
  arma::vec eigvals;
  arma::mat eigvecs;
  
  arma::eig_sym(eigvals, eigvecs, vaidya_new_hess);
  
  new_sqrt_inv_hess = eigvecs * arma::diagmat(arma::sqrt((1 / eigvals))) * eigvecs.t();
}

// [[Rcpp::export]]
arma::mat generate_vaidya_samples(const int n, int burnin, const arma::vec & initial, 
                                  const arma::mat & A, const arma::vec & b, 
                                  const double r, double lazy) {
  
  arma::mat samples(A.n_cols, n);
  VaidyaWalker vaidyaw = VaidyaWalker(initial, A, b, r);
  
  arma::vec new_sample = arma::zeros(A.n_cols);

  for (int i = 0; i < burnin; i++) {
    vaidyaw.do_sample(new_sample, lazy);
  }
    
  for (int i = 0; i < n; i++) {
    vaidyaw.do_sample(new_sample, lazy);
    samples.col(i) = new_sample;
  }
  
  return samples;
}

