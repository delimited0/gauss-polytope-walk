#ifndef JOHN_WALKER_HPP
#define JOHN_WALKER_HPP

#include "walker.hpp"

class JohnWalker : public Walker {
public:
  JohnWalker(const arma::vec & initial, const arma::mat & A, const arma::vec & b,
             const double r) : Walker(initial, A, b), r(r), 
             alpha(1. - 1. / std::log2(2. * A.n_rows / A.n_cols)),
             beta(A.n_cols / 2. / A.n_rows),
             curr_weight(arma::ones(A.n_rows)) {}
  
  double get_radius() {
    return r;
  }
  
  void proposal(arma::vec & new_sample);
  
  bool accept_reject_reverse(const arma::vec & new_sample);
  
  bool do_sample(arma::vec& new_sample, const double lazy = 0.5);
  
  void sqrt_inv_hess_barrier(const arma::vec & new_sample, arma::mat & new_sqrt_inv_hess);
  
private:
  const double r;
  const double alpha;
  const double beta;
  arma::vec curr_weight;
};

#endif