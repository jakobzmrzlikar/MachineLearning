#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <string>

// training_data vector consisting of pairs (x,y)
typedef std::vector<std::vector<double>> data;

class NeuralNetwork {

public:

    // vector of weights
    data w;
    double C;
    NeuralNetwork();

    // call function
    double h(std::vector<double>& x);

    double cost(data& training_data, std::string mode);

    // train returns vector of costs at each epoch
    std::vector<double> train(data& training_data, double learning_rate, int epochs, double C);

    void save(std::string filename="NeuralNetwork.csv");

    void load(std::string filename="NeuralNetwork.csv");

private:

    void gradient_descent(data& training_data, double learning_rate, int m);

};

std::ostream& operator<<(std::ostream& os, const NeuralNetwork& m);

#endif /* NEURALNETWORK_H */
