#ifndef SVM_H
#define SVM_H

#include <vector>
#include <string>

// training_data vector consisting of pairs (x,y)
typedef std::vector<std::vector<double>> data;

class SVM {

public:

    // vector of weights
    std::vector<double> w;
    double C;

    SVM();

    // call function
    double h(std::vector<double>& x);
    // The above function actually returns -1 or 1, double used for consistency

    double cost(data& training_data, int m);

    // train returns vector of costs at each epoch
    std::vector<double> train(data& training_data, double learning_rate, int epochs, double C);

    void save(std::string filename);

    void load(std::string filename);

private:

    void gradient_descent(data& training_data, double learning_rate, int m, double C);


};

#endif // SVM_H
