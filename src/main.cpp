#include "LinearRegression.hpp"
#include "LogisticRegression.hpp"
#include "CrossValidation.cpp"

typedef std::vector<std::vector<double>> data;

int main() {

  LogisticRegression example;

  CrossValidation(example, "shuttle", 5);

  return 0;
}
