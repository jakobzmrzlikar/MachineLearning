#ifndef COST_H
#define COST_H

class QuadraticCost {

public:
    static double cost(double a, double y);
    static double error_b(double a, double y);
    static double error_w(double a, double y, double x);

};

class CrossEntropyCost {
public:
  static double cost(double a, double y);
  static double error_b(double a, double y);
  static double error_w(double a, double y, double x);

};


#endif // COST_H
