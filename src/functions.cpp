#include <cmath>

#include "functions.h"

double QuadraticCost::cost(double a, double y) {
    return(0.5*pow((a-y), 2));
}

double QuadraticCost::error_b(double a, double y) {
    return(a-y);
}

double QuadraticCost::error_w(double a, double y, double x) {
    return((a-y)*x);
}
