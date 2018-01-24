#include <iostream>
#include <vector>
#include "LinearRegression.h"
#include "data_loader.cpp"

typedef std::vector<std::vector<double>> data;

int main() {

    double learning_rate = 0.00009;
    int epochs = 10000;

    DataLoader loader("data.csv");
    data training_data = loader.load();

    LinearRegression example(training_data[0].size());
    std::vector<double> cost = example.train(training_data, learning_rate, epochs);


    for (int i=0; i<cost.size(); i++) {
        std::cout << "cost: " << cost[i] << '\n';
    }

    return 0;
}
