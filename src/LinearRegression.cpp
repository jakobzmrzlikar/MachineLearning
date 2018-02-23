#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "LinearRegression.hpp"
#include "cost.hpp"
#include "functions.hpp"

// training_data vector consisting of pairs (x,y)
typedef std::vector<std::vector<double>> data;

LinearRegression::LinearRegression(){}

double LinearRegression::h(std::vector<double>& x) {
    double result = dot_product(w, x);
    return result;
}

double LinearRegression::cost(data& training_data, std::string mode) {
    int m = training_data.size();
    double cost = 0.0;
    for (int i=0; i<m; i++) {
        std::vector<double> x(training_data[i].begin(), training_data[i].end()-1);
        double y = training_data[i].back();
        double a = h(x);
        cost += QuadraticCost::cost(a, y);
    }

    return cost/m;
}

std::vector<double> LinearRegression::train(data& training_data, double learning_rate, int epochs, double C) {

    w.clear();
    for (int i=0; i<training_data[0].size(); i++) {
        w.push_back(0.0);
    }

    int m = training_data.size();
    double lambda = 1/(2*m*C);

    std::vector<double> training_cost;
    training_cost.push_back(cost(training_data));
    for (int i=0; i<epochs; i++){
        gradient_descent(training_data, learning_rate, m);
        training_cost.push_back(cost(training_data));
        if (training_cost[i+1] > training_cost[i]) { // early stopping
          return training_cost;
        }
    }

    return training_cost;

}

void LinearRegression::gradient_descent(data& training_data, double learning_rate, int m) {
    std::vector<double> error(training_data[0].size());

    // compute gradients
    for (int i=0; i<m; i++) {
        std::vector<double> x(training_data[i].begin(), training_data[i].end()-1);
        double y = training_data[i].back();
        double a = h(x);
        error[0] += QuadraticCost::error_b(a, y);
        for (int j=1; j<training_data[0].size(); j++) {
          error[j] += QuadraticCost::error_w(a, y, x[j-1]);
        }
    }

    // update parameters with gradients
    for (int i=0; i<w.size(); i++) {
        w[i] -= learning_rate / m * error[i];
    }

}

void LinearRegression::save(std::string filename) {
    std::string name = "../data/saves/" + filename;
    std::ofstream file(name);
    file << w[0];
    for (int i=1; i<w.size(); i++) {
      file << "," << w[i];
    }
    file << std::endl;
    std::cout << "Model saved to path: " << name << '\n';
}

void LinearRegression::load(std::string filename) {
    std::string name = "../data/saves/" + filename;
    std::ifstream file(name);
    std::string line;
    w.clear();

    while(getline(file,line)) {
        std::stringstream linestream(line);
        std::string value;

        while(getline(linestream,value,',')) {
            w.push_back(atof(value.c_str()));
        }

    }

}

std::ostream& operator<<(std::ostream& os, const LinearRegression& m) {
  return os << "Linear Regression";
}
