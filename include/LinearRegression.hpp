#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include <vector>
#include <string>

// training_data vector consisting of pairs (x,y)
typedef std::vector<std::vector<double>> data;

class LinearRegression {

public:

    // vector of weights
    std::vector<double> w;

    LinearRegression();

    // call function
    double h(std::vector<double>& x);

    double cost(data& training_data, std::string mode="regression");

    // train returns vector of costs at each epoch
    std::vector<double> train(data& training_data, double learning_rate, int epochs);

    void save(std::string filename="LinearRegression.csv");

    void load(std::string filename="LinearRegression.csv");

private:

    void gradient_descent(data& training_data, double learning_rate, int m);

};

std::ostream& operator<<(std::ostream& os, const LinearRegression& m);

#endif /* LINEARREGRESSION_H */
