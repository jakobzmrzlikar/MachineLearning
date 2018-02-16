#include "SVM.hpp"
#include "functions.hpp"

// training_data vector consisting of pairs (x,y)
typedef std::vector<std::vector<double>> data;

SVM::SVM() {}

double SVM::h(std::vector<double>& x) {
    double result = dot_product(w, x);
    if (result >= 1) {
        return 1;
    } else if (result <= -1) {
        return -1;
    } else {
        return 0;
    }
}
