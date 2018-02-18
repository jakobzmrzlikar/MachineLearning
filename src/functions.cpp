#include <cmath>
#include <algorithm>
#include "functions.hpp"

data normalize(data& training_data) {
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

data scale(data& training_data) {
  for (int i=0; i<training_data[0].size(); i++) {
    std::vector<double> feature_list;
    for (int j=0; j<training_data.size(); j++) {
      feature_list.push_back(training_data[j][i]);
    }

    double min = *min_element(feature_list.begin(), feature_list.end());
    double max = *max_element(feature_list.begin(), feature_list.end());

    for (int j=0; j<training_data.size(); j++) {

      // There could be problems here if max-min == 0, update needed!
      training_data[j][i] = (training_data[j][i]-min)/(max-min);
    }
  }

  return training_data;
}



double sigmoid(double z) {
  return 1.0/(1.0+exp(-z));
}

double sigmoid_prime(double z) {
  return sigmoid(z)*(1-sigmoid(z));
}

double dot_product(std::vector<double>& w, std::vector<double>& x) {
  double result = w[0];
  for (int i=1; i<w.size(); i++) {
      result+=w[i] * x[i-1];
  }
  return result;
}

double norm(std::vector<double> w) {
  double result = 0;
  for (int i=0; i<w.size(); i++) {
    result += pow(w[i], 2);
  }
  return sqrt(result);
}

double distance(std::vector<double> a, std::vector<double> b) {
  double result = 0;
  for (int i=0; i<a.size(); i++) {
    result += pow((a[i]-b[i]), 2);
  }
  return sqrt(result);
}
