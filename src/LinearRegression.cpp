#include <vector>
#include "LinearRegression.h"
#include "functions.h"

// training_data vector consisting of pairs (x,y)
typedef std::vector<std::vector<double>> data;

LinearRegression::LinearRegression(){
    w.push_back(0.0);
    w.push_back(0.0);
}

double LinearRegression::h(double x) {
    return (w[0] + w[1] * x);
}

std::vector<double> LinearRegression::train(data training_data, double learning_rate, int epochs) {

    int m = training_data.size();

    // normalize input to avoid overflow
    double avg=0.0;
    for (int i=0; i<m; i++) {
        avg+=training_data[i][0];
    }
    avg = avg/m;
    for (int i=0; i<m; i++) {
        training_data[i][0] -= avg;
    }

    std::vector<double> costs = {cost(training_data, m)};
    
    for (int i=0; i<epochs; i++){
        gradient_descent(training_data, learning_rate, m);
        costs.push_back(cost(training_data, m));
    }

    return costs;

}

double LinearRegression::cost(data training_data, int m) {
    int cost = 0;
    for (int i=0; i<m; i++) {
        double x = training_data[i][0];
        double y = training_data[i][1];
        double a = h(x);
        cost += QuadraticCost::cost(a, y);
    }

    return cost/m;
}

void LinearRegression::gradient_descent(data training_data, double learning_rate, int m) {
    std::vector<double> error(2);

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
        this->w[i] -= learning_rate / m * error[i];
    }
}

