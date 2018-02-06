#include <iostream>
#include <vector>
#include "functions.hpp"
#include "LinearRegression.hpp"
#include "DataLoader.hpp"

typedef std::vector<std::vector<double>> data;

int main() {

    // Hyperparameters:
    double learning_rate = 0.000001;
    int epochs = 100;

<<<<<<< HEAD
    DataLoader loader("train2.csv");
    data training_data = loader.load();

    DataLoader test("test.csv");
    data test_data = test.load();
=======
    data training_data = DataLoader::load("train.csv");
    data test_data = DataLoader::load("test.csv");
>>>>>>> f5d908ac33b99fff9c868a114771ca376e5009ff

    LinearRegression example(training_data[0].size());
    std::vector<double> training_cost = example.train(training_data, learning_rate, epochs);
    //example.save("save.csv");

    example.load("save.csv");

    double test_cost = example.cost(test_data, test_data.size());

    /*

    for (int i=0; i<training_cost.size(); i++) {
        std::cout << training_cost[i] << std::endl;
    }
    */

    //std::cout << test_cost << std::endl;


    return 0;
}
