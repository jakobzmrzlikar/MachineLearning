#ifndef FUNCTIONS_H
#define FUNCTIONS_H

class QuadraticCost {

public:
    static double cost(double a, double y);
    static double error_b(double a, double y);
    static double error_w(double a, double y, double x);

};

#endif /* FUNCTIONS_H */

