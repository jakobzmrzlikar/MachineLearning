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

class HingeLoss {
public:
  static double cost(double a, double y);
  static double error_b(double lambda);
  static double error_w(double a, double y, double x, double w, double lambda);
};

#endif // COST_H
