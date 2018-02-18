#include <cmath>
#include <algorithm>
#include "cost.hpp"

// Quadratic Cost

double QuadraticCost::cost(double a, double y) {
    return(0.5*pow((a-y), 2));
}

double QuadraticCost::error_b(double a, double y) {
    return(a-y);
}

double QuadraticCost::error_w(double a, double y, double x) {
    return((a-y)*x);
}


// Cross-Entropy Cost

double CrossEntropyCost::cost(double a, double y) {
  return(-1.0*(y*log(a)+(1.0-y)*log(1.0-a)));
}

double CrossEntropyCost::error_b(double a, double y) {
  return (a-y);
}

double CrossEntropyCost::error_w(double a, double y, double x) {
    return((a-y)*x);
}

double HingeLoss::cost(double a, double y) {
  return std::max(0.0, 1-y*a);
}

double HingeLoss::error_b(double lambda) {
  return lambda;
}

double HingeLoss::error_w(double a, double y, double x, double w, double lambda) {
  if (y*a < 1) {
    return (lambda*w - y*x);
  } else {
    return lambda*w;
  }
}
