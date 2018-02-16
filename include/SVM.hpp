#ifndef SVM_H
#define SVM_H

#include <vector>

// training_data vector consisting of pairs (x,y)
typedef std::vector<std::vector<double>> data;

class SVM {

public:

    // vector of weights
    std::vector<double> w;

    SVM();

    // call function
    double h(std::vector<double>& x);
    // The above function actually returns -1 or 1, double used for consistency




};

#endif // SVM_H
