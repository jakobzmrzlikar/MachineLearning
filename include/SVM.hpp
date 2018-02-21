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

    double cost(data& training_data, std::string mode);

    // train returns vector of costs at each epoch
    std::vector<double> train(data& training_data, double learning_rate, int epochs, double C);

    void save(std::string filename="SVM.csv");

    void load(std::string filename="SVM.csv");

private:

    void gradient_descent(data& training_data, double learning_rate, int m, double C);


};

std::ostream& operator<<(std::ostream& os, const SVM& m);

#endif // SVM_H
