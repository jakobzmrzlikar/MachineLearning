#include <vector>
#include <string>
#include <iostream>
#include <ctime>
#include <chrono>

#include "DataLoader.hpp"
#include "functions.hpp"

typedef std::vector<std::vector<double>> data;

template <typename T>
void test(T& model, std::string dataset, std::string mode) {

  model.load();

  std::string data_name = "../data/" + dataset + "/test.csv";
  data test_data = load(data_name);

  // Optional data scaling
  test_data = scale(test_data);

  double cost = model.cost(test_data, mode);

  std::time_t end_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

  // Report
  std::cout << "Date: " << std::ctime(&end_time) << '\n';
  std::cout << "Dataset: " << dataset << '\n';
  std::cout << "Model: " << model << '\n';
  if (mode == "regression") {
    std::cout << "Test cost: " << cost << '\n';
  } else if (mode == "classification") {
    std::cout << "Test classification accuracy: " << cost << '%' << '\n';
  }

}
