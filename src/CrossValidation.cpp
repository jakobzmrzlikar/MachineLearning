#include <vector>
#include <iostream>
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
void cross_validate(T& model, std::string dataset, int k, std::string mode) {

  auto start = std::chrono::system_clock::now();

  // Hyperparameters:
  int epochs = 100;
  double learning_rate = 1e-8;

  std::string data_name = "../data/" + dataset + "/train.csv";

  data complete_data = load(data_name);

  // Optional data scaling
  complete_data = scale(complete_data);

  // K-fold Cross Validation algorithm
  std::vector<double> training_cost;
  std::vector<double> validation_cost;
  data training_data;
  data validation_data;

  int stride = floor(complete_data.size()/k);
  for (int i=0; i<k; i++) {
    training_data.clear();
    validation_data.clear();

    for (int j=0; j<complete_data.size(); j++) {
      if ((j >= i*stride) and (j < (i+1)*stride)) {
        validation_data.push_back(complete_data[j]);
      } else {
        training_data.push_back(complete_data[j]);
      }
    }

    std::random_shuffle(training_data.begin(), training_data.end());
    std::random_shuffle(validation_data.begin(), validation_data.end());

    training_cost = model.train(training_data, learning_rate, epochs, model.C);
    validation_cost.push_back(model.cost(validation_data, "training"));

    learning_rate *= 10;
  }

  // Replicate best case scenaro
  double min = validation_cost[0];
  for (int i=1; i<validation_cost.size(); i++) {
    if (validation_cost[i] > 0 and validation_cost[i] < min) {
      min = validation_cost[i];
    }
  }
  int index = std::distance(validation_cost.begin(), find(validation_cost.begin(), validation_cost.end(), min));
  learning_rate = 1e-8 * pow(10, index);
  epochs = 1000;

  model.w.clear();
  training_cost = model.train(training_data, learning_rate, epochs, model.C);
  validation_cost.push_back(model.cost(validation_data, "training"));


  auto stop = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = stop-start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(stop);

  // Report
  std::cout << "CROSS VALIDATION" << '\n';
  std::cout << "Date: " << std::ctime(&end_time)<< '\n';
  std::cout << "Dataset: " << dataset << '\n';
  std::cout << "Model: " << model << '\n';
  std::cout << "Mode: " << mode << '\n';
  std::cout << "Scaling: Yes" << '\n';
  std::cout << "Cross Validation batches: " << k << '\n';
  std::cout << "Epochs: " << epochs << '\n';
  std::cout << "Learning rate: " << learning_rate << '\n';
  std::cout << "Regularization: " << model.C << '\n';
  std::cout << "Final training cost: " << training_cost.back() << '\n';
  std::cout << "Final validation cost: " << validation_cost.back() << '\n';
  std::cout << "Time running: " << elapsed_seconds.count() << " seconds" << '\n';

  model.save();

  std::cout << "--------------------------------------------------------------------------------" << '\n';


  std::string name = "../data/" + dataset + "/training_cost.csv";
  std::ofstream file(name);
  for (int i=0; i<training_cost.size(); i++) {
    file << training_cost[i] << '\n';
  }

  std::string name2 = "../data/" + dataset + "/validation_cost.csv";
  std::ofstream file2(name2);
  for (int i=0; i<validation_cost.size()-1; i++) {
    file2 << validation_cost[i] << '\n';
  }

}

template <class T>
void cross_validate_KNN(T& model, std::string dataset, int k, std::string mode) {

  auto start = std::chrono::system_clock::now();

  // Hyperparameters:
  model.K = 1;

  std::string data_name = "../data/" + dataset + "/train.csv";

  data complete_data = load(data_name);

  // Optional data scaling
  complete_data = scale(complete_data);

  std::vector<double> training_cost;
  std::vector<double> validation_cost;
  data training_data;
  data validation_data;

  int stride = complete_data.size()*0.3;
  for (int j=0; j<complete_data.size(); j++) {
    if (j < stride) {
      validation_data.push_back(complete_data[j]);
    } else {
      training_data.push_back(complete_data[j]);
    }
  }
  model.train(training_data);
  validation_cost.push_back(0);

  // // Cross validation
  // for (int i=0; i<k; i++) {
  //   validation_cost.push_back(model.cost(validation_data, mode));
  //   model.K += 2;
  // }
  //
  // // Replicate best case scenaro
  // double min = validation_cost[0];
  // for (int i=1; i<validation_cost.size(); i++) {
  //   if (std::isfinite(validation_cost[i]) and validation_cost[i] < min) {
  //     min = validation_cost[i];
  //   }
  // }
  // int index = std::distance(validation_cost.begin(), find(validation_cost.begin(), validation_cost.end(), min));
  // //model.K = 1 + 2 * index;
  //
  // model.train(training_data);
  // validation_cost.push_back(model.cost(validation_data, mode));

  auto stop = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = stop-start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(stop);

  // Report
  std::cout << "CROSS VALIDATION" << '\n';
  std::cout << "Date: " << std::ctime(&end_time)<< '\n';
  std::cout << "Dataset: " << dataset << '\n';
  std::cout << "Model: " << model << '\n';
  std::cout << "Mode: " << mode << '\n';
  std::cout << "Scaling: Yes" << '\n';
  std::cout << "Cross Validation batches: " << k << '\n';
  std::cout << "K: " << model.K << '\n';
  std::cout << "Final validation cost: " << validation_cost.back() << '\n';
  std::cout << "Time running: " << elapsed_seconds.count() << " seconds" << '\n';

  model.save();

  std::cout << "--------------------------------------------------------------------------------" << '\n';

  std::string name2 = "../data/" + dataset + "/validation_cost.csv";
  std::ofstream file2(name2);
  for (int i=0; i<validation_cost.size()-1; i++) {
    file2 << validation_cost[i] << '\n';
  }

}

template <class T>
void cross_validate_NB(T& model, std::string dataset) {
  auto start = std::chrono::system_clock::now();

  std::string data_name = "../data/" + dataset + "/train.csv";
  data complete_data = load(data_name);

  // Optional data scaling
  complete_data = scale(complete_data);

  model.train(complete_data);

  auto stop = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = stop-start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(stop);

  // Report
  std::cout << "CROSS VALIDATION" << '\n';
  std::cout << "Date: " << std::ctime(&end_time)<< '\n';
  std::cout << "Dataset: " << dataset << '\n';
  std::cout << "Model: " << model << '\n';
  std::cout << "Scaling: Yes" << '\n';
  std::cout << "Time running: " << elapsed_seconds.count() << " seconds" << '\n';

  model.save();

  std::cout << "--------------------------------------------------------------------------------" << '\n';

}
