#include <vector>
#include <iostream>
#include "LinearRegression.hpp"
#include "functions.hpp"

// training_data vector consisting of pairs (x,y)
typedef std::vector<std::vector<double>> data;

LinearRegression::LinearRegression(int features){
    for (int i=0; i<features; i++) {
        w.push_back(0.0);
    }

}

double LinearRegression::h(double x) {
    return (w[0] + w[1] * x);
}

std::vector<double> LinearRegression::train(data training_data, double learning_rate, int epochs) {

    int m = training_data.size();

    std::vector<double> costs;
    costs.push_back(cost(training_data, m));

    for (int i=0; i<epochs; i++){
        gradient_descent(training_data, learning_rate, m);
        costs.push_back(cost(training_data, m));
    }

    return costs;

}

double LinearRegression::cost(data training_data, int m) {
    double cost = 0.0;
    for (int i=0; i<m; i++) {
        double x = training_data[i][0];
        double y = training_data[i][1];
        double a = h(x);
        cost += QuadraticCost::cost(a, y);
    }

    return cost/m;
}

void LinearRegression::gradient_descent(data training_data, double learning_rate, int m) {
    std::vector<double> error(training_data[0].size());

    // compute gradients
    for (int i=0; i<m; i++) {
        double x = training_data[i][0];
        double y = training_data[i][1];
        double a = h(x);
        error[0] += QuadraticCost::error_b(a, y);
        error[1] += QuadraticCost::error_w(a, y, x);
    }


    // update parameters with gradients
    for (int i=0; i<this->w.size(); i++) {
        w[i] -= learning_rate / m * error[i];
    }
}
