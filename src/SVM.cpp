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
    if (result >= 0) {
        return 1;
    } else if (result < 0) {
        return -1;
    }
}

double SVM::cost(data& training_data, std::string mode) {
    int m = training_data.size();
    double cost = 0.0;
    double correct_class = 0;
    for (int i=0; i<m; i++) {
        std::vector<double> x(training_data[i].begin(), training_data[i].end()-1);
        double y = training_data[i].back();
        if (y==0) y = -1;
        double a = h(x);
        cost += HingeLoss::cost(a, y);
        if (fabs(a-y)<0.5) correct_class++;
    }

    if (mode == "training") {
      return cost+pow(norm(w), 2);
    } else if (mode == "classification") {
      return correct_class/m * 100;
    }
}

std::vector<double> SVM::train(data& training_data, double learning_rate, int epochs, double C) {

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
        if (training_cost[i+1] < training_cost[i]) { // early stopping
          return training_cost;
        }
    }

    return training_cost;
  }

  void SVM::gradient_descent(data& training_data, double learning_rate, int m, double C) {
    std::vector<double> error(training_data[0].size());

    // compute gradients
    for (int i=0; i<m; i++) {
        std::vector<double> x(training_data[i].begin(), training_data[i].end()-1);
        double y = training_data[i].back();
        if (y==0) y = -1;
        double a = dot_product(w, x);
        double lambda = 1/(2*m*C);
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
      std::string name = "../data/saves/" + filename;
      std::ofstream file(name);
      file << w[0];
      for (int i=1; i<w.size(); i++) {
        file << "," << w[i];
      }
      file << std::endl;
      std::cout << "Model saved to path: " << name << '\n';
  }

  void SVM::load(std::string filename) {
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

std::ostream& operator<<(std::ostream& os, const SVM& m) {
  return os << "SVM";
}
