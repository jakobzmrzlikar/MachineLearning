#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <math.h>
#include "NeuralNetwork.hpp"
#include "cost.hpp"
#include "functions.hpp"

// training_data vector consisting of pairs (x,y)
typedef std::vector<std::vector<double>> data;

NeuralNetwork::NeuralNetwork(){}

double NeuralNetwork::h(std::vector<double>& x) {
    std::vector<double> result;
    std::vector<double> z;
    std::vector<double> a = x;
    for (int i=0; i<w.size(); i++) {
      a = dot_product(w[i], x);
      x = sigmoid(a);
      z.push_back(x);

    }
    result = z;
    return result;
}
