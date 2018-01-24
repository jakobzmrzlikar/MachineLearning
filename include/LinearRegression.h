#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include <vector>

// training_data vector consisting of pairs (x,y)
typedef std::vector<std::vector<double>> data;

class LinearRegression {

public:

    // vector of weights
    std::vector<double> w;

    LinearRegression(int features);

    // call function
    double h(double x);

    // train returns vector of costs at each epoch
    std::vector<double> train(data training_data, double learning_rate, int epochs);

private:

    double cost(data training_data, int m);

    void gradient_descent(data training_data, double learning_rate, int m);

};

#endif /* LINEARREGRESSION_H */
