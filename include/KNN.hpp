#ifndef KNN_H
#define KNN_H

#include <vector>

typedef std::vector<std::vector<double>> data;

class KNN {
public:

  KNN();

  // call function
  double h(std::vector<double>& x, int k);

  void train(data& training_data);

private:
  data space;

  std::vector<double> kNeighbours(std::vector<double>& x, int k);
};

#endif // KNN_H
