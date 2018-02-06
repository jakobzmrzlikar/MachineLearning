#include <iostream>
#include <vector>
#include "functions.hpp"
#include "LinearRegression.hpp"
#include "DataLoader.hpp"

typedef std::vector<std::vector<double>> data;

int main() {

    double learning_rate = 0.0001;
    int epochs = 100;

    data training_data = DataLoader::load("train.csv");
    data test_data = DataLoader::load("test.csv");

    LinearRegression example(training_data[0].size());
    std::vector<double> training_cost = example.train(training_data, learning_rate, epochs);
    example.save("save.csv");


    double test_cost = 0.0;
    double a = 0;
    double y = 0;
    for (int i=0; i<test_data.size(); i++) {
        a = example.h(test_data[i][0]);
        y = test_data[i][1];
        test_cost += QuadraticCost::cost(a, y);
        //std::cout << a << " " << y << std::endl;
    }
    test_cost = test_cost/test_data.size();

    /*
    for (int i=0; i<training_cost.size(); i++) {
        std::cout << training_cost[i] << std::endl;
    }

    std::cout << test_cost << std::endl;
    */

    return 0;
}
