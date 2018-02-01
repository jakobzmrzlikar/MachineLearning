#include <iostream>
#include <vector>
#include "LinearRegression.hpp"
#include "data_loader.cpp"

typedef std::vector<std::vector<double>> data;

int main() {

    double learning_rate = 0.0001;
    int epochs = 100;

    DataLoader loader("train.csv");
    data training_data = loader.load();
    DataLoader test("test.csv");
    data test_data = test.load();

    LinearRegression example(training_data[0].size());
    std::vector<double> training_cost = example.train(training_data, learning_rate, epochs);


    double test_cost = 0.0;
    double a = 0;
    double y = 0;
    for (int i=0; i<test_data.size(); i++) {
        a = example.h(test_data[i][0]);
        y = test_data[i][1];
        test_cost += QuadraticCost::cost(a, y);
        //std::cout << a << " " << y << '\n';
    }
    test_cost = test_cost/test_data.size();

    for (int i=0; i<training_cost.size(); i++) {
      std::cout << training_cost[i] << '\n';
    }

    std::cout << test_cost << '\n';

    return 0;
}
