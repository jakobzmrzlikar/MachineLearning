#include <iostream>
#include <vector>
#include "LinearRegression.h"
#include "data_loader.cpp"

typedef std::vector<std::vector<double>> data;

int main() {

    double learning_rate = 0.1;
    int epochs = 100;

    LinearRegression example;
    DataLoader loader("data.csv");
    data training_data = loader.load();
    std::vector<double> cost = example.train(training_data, learning_rate, epochs);

    /*
    for (int i=0; i<cost.size(); i++) {
        std::cout << cost[i] << '\n';
    }
    */

    for (int i=2012; i<2020; i++) {
        std::cout << i << " " << example.h(i) << '\n';
    }

    return 0;
}

