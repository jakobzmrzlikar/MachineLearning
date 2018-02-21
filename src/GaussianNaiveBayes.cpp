#define _USE_MATH_DEFINES // for using M_PI constant from <cmath>

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include "NaiveBayes.hpp"


// training_data vector consisting of pairs (x,y)
typedef std::vector<std::vector<double>> data;

GaussianNaiveBayes::GaussianNaiveBayes() {}

double GaussianNaiveBayes::h(std::vector<double>& x) {
  double label;
  double probability = 0;
  double data_probability = 0; // P(d)
  for (int i=0; i<class_frequencies.size(); i++) {
    double new_probability = class_frequencies[i]/class_frequencies.size();
    // P(class==i)
    for (int j=0; j<x.size(); j++) {
      new_probability *= pdf(x[j], means[j], standard_deviations[j])/class_frequencies[i];
      // P(pdf(x[j]) | class==i)

    }

    data_probability += new_probability;
    if (new_probability > probability) {
      probability = new_probability;
      label = i;
    }
  }
  probability /= data_probability;
  // Calculate final probability, even though label is known before that.
  std::cout << "Probability: " << probability*100 << '%' << '\n';

  return label;
}

// train DOESN'T WORK for negative labels!
void GaussianNaiveBayes::train(data& training_data) {
  double label;
  means.resize(training_data[0].size());
  standard_deviations.resize(training_data[0].size());

  // Calculate means and class frequencies
  for (int i=0; i<training_data.size(); i++) {
    label = training_data[i].back();
    if (label >= class_frequencies.size()) {
      class_frequencies.resize(label+1);
    }
    class_frequencies[label]++;

    for(int j=0; j<training_data[0].size()-1; j++) {
      means[j] += training_data[i][j]/training_data.size();
    }

  }

  // Calculate standard deviations
  for (int i=0; i<training_data.size(); i++) {
    for (int j=0; j<training_data[0].size()-1; j++) {
      standard_deviations[j] += pow((training_data[i][j]-means[j]), 2)/training_data.size();
    }

    for (int j=0; j<standard_deviations.size(); j++) {
      standard_deviations[j] = sqrt(standard_deviations[j]);
    }
  }

}

double GaussianNaiveBayes::cost(data& training_data, std::string mode) {
  int m = training_data.size();
  int correct_class = 0;
  for (int i=0; i<m; i++) {
      std::vector<double> x(training_data[i].begin(), training_data[i].end()-1);
      double y = training_data[i].back();
      double a = h(x);
      if (fabs(a-y)<0.5) correct_class++;
  }

  if (mode == "classfiaction") {
    return correct_class/m * 100;
  }
}

double GaussianNaiveBayes::pdf(double x, double mean, double sd) {
  return (1/(sqrt(2*M_PI)*sd)) * exp(-((x-pow(mean, 2))/(2*pow(sd, 2))));
}

void GaussianNaiveBayes::save(std::string filename) {
    std::string name = "../data/saves/" + filename;
    std::ofstream file(name);

    file << class_frequencies[0];
    for (int i=1; i<class_frequencies.size(); i++) {
      file << "," << class_frequencies[i];
    }
    file << '\n';

    file << means[0];
    for (int i=1; i<means.size(); i++) {
      file << "," << means[i];
    }
    file << '\n';

    file << standard_deviations[0];
    for (int i=1; i<standard_deviations.size(); i++) {
      file << "," << standard_deviations[i];
    }
    file << '\n';

    std::cout << "Model saved to path: " << name << '\n';
}

void GaussianNaiveBayes::load(std::string filename) {
    std::string name = "../data/saves" + filename;
    std::ifstream file(name);
    std::string line;
    class_frequencies.clear();
    means.clear();
    standard_deviations.clear();
    int i=0;

    while(getline(file,line)) {
        std::stringstream linestream(line);
        std::string value;

        if  (i==0) {
          while(getline(linestream,value,',')) {
              class_frequencies.push_back(atof(value.c_str()));
          }
        } else if (i==1) {
          while(getline(linestream,value,',')) {
              means.push_back(atof(value.c_str()));
          }
        } else if (i==2) {
          while(getline(linestream,value,',')) {
              standard_deviations.push_back(atof(value.c_str()));
          }
        }
        i++;

    }
}

std::ostream& operator<<(std::ostream& os, const GaussianNaiveBayes& m) {
  return os << "Gaussian Naive Bayes";
}
