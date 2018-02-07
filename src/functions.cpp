#include <cmath>
#include <algorithm>
#include "functions.hpp"

typedef std::vector<std::vector<double>> data;


// Quadratic Cost

double QuadraticCost::cost(double a, double y) {
    return(0.5*pow((a-y), 2));
}

double QuadraticCost::error_b(double a, double y) {
    return(a-y);
}

double QuadraticCost::error_w(double a, double y, double x) {
    return((a-y)*x);
}


// Cross-Entropy Cost

double CrossEntropyCost::cost(double a, double y) {
  return(-1.0*(y*log(a)+(1.0-y)*log(1.0-a)));
}

double CrossEntropyCost::error_b(double a, double y) {
  return (a-y);
}

double CrossEntropyCost::error_w(double a, double y, double x) {
    return((a-y)*x);
}


// Standardization

data Standardization::normalize(data training_data) {
  int m = training_data.size();

  for (int i=0; i<training_data[0].size()-1; i++){
    double avg=0.0;
    for (int j=0; j<m; j++) {
        avg+=training_data[j][i];
    }

    avg = avg/m;

    for (int j=0; j<m; j++) {
        training_data[j][i] -= avg;
    }
  }
  return training_data;
}

data Standardization::scale(data training_data) {
  for (int i=0; i<training_data[0].size(); i++) {
    std::vector<double> feature_list;
    for (int j=0; j <training_data.size(); j++) {
      feature_list.push_back(training_data[j][i]);
    }

    double min = *min_element(feature_list.begin(), feature_list.end());
    double max = *max_element(feature_list.begin(), feature_list.end());

    for (int j=0; j<training_data.size(); j++) {
      training_data[j][i] = (training_data[j][i]-min)/(max-min);
    }
  }

  return training_data;
}


// Functions

double Functions::sigmoid(double z) {
  return 1.0/(1.0+exp(-z));
}

double Functions::sigmoid_prime(double z) {
  return Functions::sigmoid(z)*(1-Functions::sigmoid(z));
}
