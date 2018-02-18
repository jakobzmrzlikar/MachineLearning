#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <chrono>
#include <string>
#include <fstream>
#include "functions.hpp"
#include "DataLoader.hpp"

typedef std::vector<std::vector<double>> data;

template <class T>
void CrossValidation(T& model, std::string dataset, int k) {

  auto start = std::chrono::system_clock::now();

  // Hyperparameters:
  double learning_rate = 0.01;
  int epochs = 1000;
  double C = 10;
  int K = 3;

  std::string data_name = "../data/" + dataset + "/data.csv";
  std::string save_name = dataset + "/save.csv";

  data complete_data = load(data_name);

  // Optional data scaling
  complete_data = scale(complete_data);

  // K-fold Cross Validation algorithm
  data training_cost;
  std::vector<double> test_cost;
  data training_data;
  data test_data;

  int stride = floor(complete_data.size()/k);
  for (int i=0; i<k; i++) {
    training_data.clear();
    test_data.clear();

    for (int j=0; j<complete_data.size(); j++) {
      if ((j >= i*stride) and (j < (i+1)*stride)) {
        test_data.push_back(complete_data[j]);
      } else {
        training_data.push_back(complete_data[j]);
      }
    }

    std::random_shuffle(training_data.begin(), training_data.end());
    std::random_shuffle(test_data.begin(), test_data.end());

    training_cost.push_back(model.train(training_data, learning_rate, epochs));
    test_cost.push_back(model.cost(test_data, test_data.size()));
  }

  double count = 0;
  for (int i=0; i<test_data.size(); i++) {
    std::vector<double> x(test_data[i].begin(), test_data[i].end()-1);
    double y = test_data[i].back();
    double a = model.h(x);
    // if (fabs(a-y) < 0.5) {
    //   count++;
    // }
    if ((a>0 and y>0) or (a<0 and y<0)) {
      count++;
    }
  }

  auto stop = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = stop-start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(stop);

  // Report
  std::cout << "Date: " << std::ctime(&end_time)<< '\n';
  std::cout << "Dataset: " << dataset << '\n';
  std::cout << "Model: Logistic Regression" << '\n';
  std::cout << "Cost: Cross-Entropy Cost" << '\n';
  std::cout << "Scaling: Yes" << '\n';
  std::cout << "Learning rate: " << learning_rate << '\n';
  std::cout << "Epochs: " << epochs << '\n';
  std::cout << "Regularization: " << C << '\n';
  std::cout << "Cross Validation batches: " << k << '\n';
  std::cout << "Final test classification accuracy: " << 100*count/test_data.size() << '%' << '\n';
  std::cout << "Final training cost: " << training_cost.back().back() << '\n';
  std::cout << "Final test cost: " << test_cost.back() << '\n';
  std::cout << "Time running: " << elapsed_seconds.count() << " seconds" << '\n';

  model.save(save_name);

  std::cout << "--------------------------------------------------------------------------------" << '\n';


  std::string name = "../data/" + dataset + "/training_cost.csv";
  std::ofstream file(name);
  for (int i=0; i<training_cost.size(); i++) {
    for (int j=0; j<training_cost[0].size(); j++) {
      file << training_cost[i][j] << '\n';
    }
  }

  std::string name2 = "../data/" + dataset + "/test_cost.csv";
  std::ofstream file2(name2);
  for (int i=0; i<test_cost.size(); i++) {
    file2 << test_cost[i] << '\n';
  }

}
