#include <iostream>
#include <vector>
#include "LinearRegression.hpp"
#include "functions.hpp"
#include "data_loader.cpp"

typedef std::vector<std::vector<double>> data;

int main() {

    // Hyperparameters:
    double learning_rate = 0.000001;
    int epochs = 100;

    DataLoader loader("train2.csv");
    data training_data = loader.load();

    DataLoader test("test.csv");
    data test_data = test.load();

    LinearRegression example(training_data[0].size());
    std::vector<double> training_cost = example.train(training_data, learning_rate, epochs);
    //example.save("save.csv");

    example.load("save.csv");

    double test_cost = example.cost(test_data, test_data.size());


    for (int i=0; i<training_cost.size(); i++) {
        std::cout << training_cost[i] << std::endl;
    }

    //std::cout << test_cost << std::endl;


    return 0;
}
