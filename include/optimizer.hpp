#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>

typedef std::vector<std::vector<double>> data;

template <typename T>
void gradient_descent(T& model, data& training_data, double learning_rate, int m);

#endif //OPTIMIZER_H
