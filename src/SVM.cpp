#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "SVM.hpp"
#include "functions.hpp"
#include "cost.hpp"

// training_data vector consisting of pairs (x,y)
typedef std::vector<std::vector<double>> data;

SVM::SVM() {}

double SVM::h(std::vector<double>& x) {
    double result = dot_product(w, x);
    // if (result >= 1) {
    //     return 1;
    // } else if (result <= -1) {
    //     return -1;
    // } else {
    //     return 0;
    // }
}

double SVM::cost(data& training_data, int m) {
    double cost = 0.0;
    for (int i=0; i<m; i++) {
        std::vector<double> x(training_data[i].begin(), training_data[i].end()-1);
        double y = training_data[i].back();
        double a = h(x);
        cost += HingeLoss::cost(a, y);
    }

    return cost+pow(norm(w), 2);
}

std::vector<double> SVM::train(data& training_data, double learning_rate, int epochs, double C) {

    if (w.empty()) {
      for (int i=0; i<training_data[0].size(); i++) {
          w.push_back(0.0);
      }
    }

    int m = training_data.size();

    std::vector<double> training_cost;
    training_cost.push_back(cost(training_data, m));
    for (int i=0; i<epochs; i++){
        gradient_descent(training_data, learning_rate, m, C);
        training_cost.push_back(cost(training_data, m));
    }

    return training_cost;
  }

  void SVM::gradient_descent(data& training_data, double learning_rate, int m, double C) {
    std::vector<double> error(training_data[0].size());

    // compute gradients
    for (int i=0; i<m; i++) {
        std::vector<double> x(training_data[i].begin(), training_data[i].end()-1);
        double y = training_data[i].back();
        double a = h(x);
        double lambda = 2/(m*C);
        error[0] += HingeLoss::error_b(lambda);
        for (int j=1; j<training_data[0].size(); j++) {
          error[j] += HingeLoss::error_w(a, y, x[j-1], w[j], lambda);
        }
    }

    // update parameters with gradients
    for (int i=0; i<w.size(); i++) {
        w[i] -= learning_rate * error[i];
    }

  }

  void SVM::save(std::string filename) {
      std::string name = "../data/" + filename;
      std::ofstream file(name);
      file << w[0];
      for (int i=1; i<w.size(); i++) {
        file << "," << w[i];
      }
      file << std::endl;
      std::cout << "Model saved to path: " << name << '\n';
  }

  void SVM::load(std::string filename) {
      std::string name = "../data/" + filename;
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
