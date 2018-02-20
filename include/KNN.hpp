#ifndef KNN_H
#define KNN_H

#include <vector>
#include <string>

typedef std::vector<std::vector<double>> data;

class KNN {
public:

  KNN();

  // call function
  double h(std::vector<double>& x, int k, std::string mode);

  void train(data& training_data);
  
  void save(std::string filename);
  void load(std::string filename);

private:
  data space;

  std::vector<double> kNeighbours(std::vector<double>& x, int k);
};

#endif // KNN_H
