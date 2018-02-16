#include <cmath>
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
