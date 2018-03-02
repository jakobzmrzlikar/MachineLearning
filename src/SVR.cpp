#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "SVR.hpp"
#include "functions.hpp"
#include "cost.hpp"

// training_data vector consisting of pairs (x,y)
typedef std::vector<std::vector<double>> data;

SVR::SVR() {}

double SVR::h(std::vector<double>& x) {
    double result = dot_product(w, x);
    return result;
}

double SVR::cost(data& training_data, std::string mode) {
    int m = training_data.size();
    double cost = 0.0;
    for (int i=0; i<m; i++) {
        std::vector<double> x(training_data[i].begin(), training_data[i].end()-1);
        double y = training_data[i].back();
        double a = h(x);
        cost += QuadraticCost::cost(a, y);
    }

    if (mode == "training") {
      return C*cost/m + 0.5*pow(norm(w), 2);
    } else if (mode == "regression") {
      return cost/m;
    }
}

std::vector<double> SVR::train(data& training_data, double learning_rate, int epochs, double C) {

    w.clear();
    for (int i=0; i<training_data[0].size(); i++) {
        w.push_back(0.0);
    }

    int m = training_data.size();

    std::vector<double> training_cost;
    training_cost.push_back(cost(training_data, "training"));
    for (int i=0; i<epochs; i++){
        gradient_descent(training_data, learning_rate, m, C);
        training_cost.push_back(cost(training_data, "training"));
        if (training_cost[i+1] > training_cost[i]) { // early stopping
          return training_cost;
        }
    }

    return training_cost;
  }

  void SVR::gradient_descent(data& training_data, double learning_rate, int m, double C) {
    std::vector<double> error(training_data[0].size());

    // compute gradients
    for (int i=0; i<m; i++) {
        std::vector<double> x(training_data[i].begin(), training_data[i].end()-1);
        double y = training_data[i].back();
        double a = dot_product(w, x);
        double lambda = 1/(2*m*C);
        error[0] += QuadraticCost::error_b(a, y);
        for (int j=1; j<training_data[0].size(); j++) {
          error[j] += QuadraticCost::error_w(a, y, x[j-1]) + lambda*w[j];
          //the +lambda*w[j] term is the derivative of ||w||^2
        }
    }

    // update parameters with gradients
    for (int i=0; i<w.size(); i++) {
        w[i] -= learning_rate * error[i];
    }

  }

  void SVR::save(std::string filename) {
      std::string name = "../data/saves/" + filename;
      std::ofstream file(name);
      file << w[0];
      for (int i=1; i<w.size(); i++) {
        file << "," << w[i];
      }
      file << std::endl;
      std::cout << "Model saved to path: " << name << '\n';
  }

  void SVR::load(std::string filename) {
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

std::ostream& operator<<(std::ostream& os, const SVR& m) {
  return os << "SVR";
}
