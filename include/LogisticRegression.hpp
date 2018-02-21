#ifndef LOGISTICREGRESSION_H
#define LOGISTICREGRESSION_H

#include <vector>
#include <string>

// training_data vector consisting of pairs (x,y)
typedef std::vector<std::vector<double>> data;

class LogisticRegression {

public:

    // vector of weights
    std::vector<double> w;

    LogisticRegression();

    // call function
    double h(std::vector<double>& x);

    double cost(data& training_data, std::string mode);

    // train returns vector of costs at each epoch
    std::vector<double> train(data& training_data, double learning_rate, int epochs);

    void save(std::string filename="LogisticRegression.csv");

    void load(std::string filename="LogisticRegression.csv");

private:

    void gradient_descent(data& training_data, double learning_rate, int m);

};

std::ostream& operator<<(std::ostream& os, const LogisticRegression& m);

#endif /* LOGISTICREGRESSION_H */
