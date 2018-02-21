#ifndef NAIVEBAYES_H
#define NAIVEBAYES_H

#include <vector>
#include <string>

typedef std::vector<std::vector<double>> data;

class NaiveBayes {
public:

  NaiveBayes();

  // call function
  double h(std::vector<double>& x);

  void train(data& training_data);

  void save(std::string filename="NaiveBayes.csv");
  void load(std::string filename="NaiveBayes.csv");

private:
  data probabilities;

};

class GaussianNaiveBayes {
public:

  GaussianNaiveBayes();

  // call function
  double h(std::vector<double>& x);

  void train(data& training_data);

  void save(std::string filename="GaussianNaiveBayes.csv");
  void load(std::string filename="GaussianNaiveBayes.csv");

private:
  std::vector<double> class_frequencies;
  std::vector<double> means;
  std::vector<double> standard_deviations;

  double pdf(double x, double mean, double sd);

};

std::ostream& operator<<(std::ostream& os, const NaiveBayes& m);
std::ostream& operator<<(std::ostream& os, const GaussianNaiveBayes& m);


#endif // NAIVEBAYES_H
