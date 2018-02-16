#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <vector>

typedef std::vector<std::vector<double>> data;

data normalize(data& training_data);
data scale(data& training_data);

double sigmoid(double z);
double sigmoid_prime(double z);
double dot_product(std::vector<double>& w, std::vector<double>& x);

#endif /* FUNCTIONS_H */
