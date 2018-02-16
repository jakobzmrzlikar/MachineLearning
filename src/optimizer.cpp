#include <vector>
#include "optimizer.hpp"
#include "cost.hpp"

typedef std::vector<std::vector<double>> data;

template <typename T>
void gradient_descent(T& model, data& training_data, double learning_rate, int m) {
    std::vector<double> error(training_data[0].size());

    // compute gradients
    for (int i=0; i<m; i++) {
        std::vector<double> x(training_data[i].begin(), training_data[i].end()-1);
        double y = training_data[i].back();
        double a = model.h(x);
        error[0] += QuadraticCost::error_b(a, y);
        for (int j=1; j<training_data[0].size(); j++) {
          error[j] += QuadraticCost::error_w(a, y, x[j-1]);
        }
    }

    // update parameters with gradients
    for (int i=0; i<model.w.size(); i++) {
        model.w[i] -= learning_rate / m * error[i];
    }
}
