#ifndef VAIDYA_WALKER_HPP
#define VAIDYA_WALKER_HPP

#include "walker.hpp"

class VaidyaWalker : public Walker {
public:
  VaidyaWalker(const arma::vec & initial, const arma::mat & A, const arma::vec & b,
               const double r) : Walker(initial, A, b), r(r) {}
  
  double get_radius() {
    return r;
  }
  
  void proposal(arma::vec & new_sample);
  
  bool accept_reject_reverse(const arma::vec & new_sample);
  
  bool do_sample(arma::vec& new_sample, const double lazy = 0.5);
  
  void sqrt_inv_hess_barrier(const arma::vec & new_sample, arma::mat & new_sqrt_inv_hess);
  
private:
  const double r;
};

#endif