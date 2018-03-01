#ifndef SVR_H
#define SVR_H

#include <vector>
#include <string>

// training_data vector consisting of pairs (x,y)
typedef std::vector<std::vector<double>> data;

class SVR {

public:

    // vector of weights
    std::vector<double> w;
    double C;

    SVR();

    // call function
    double h(std::vector<double>& x);

    double cost(data& training_data, std::string mode);

    // train returns vector of costs at each epoch
    std::vector<double> train(data& training_data, double learning_rate, int epochs, double C);

    void save(std::string filename="SVR.csv");

    void load(std::string filename="SVR.csv");

private:

    void gradient_descent(data& training_data, double learning_rate, int m, double C);


};

std::ostream& operator<<(std::ostream& os, const SVR& m);

#endif // SVR_H
