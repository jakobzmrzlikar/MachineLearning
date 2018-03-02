#include "LinearRegression.hpp"
#include "LogisticRegression.hpp"
#include "SVM.hpp"
#include "SVR.hpp"
#include "KNN.hpp"
#include "NaiveBayes.hpp"
#include "CrossValidation.cpp"
#include "test.cpp"

#include <string>
#include <iostream>

int main() {

  LinearRegression lin;
  LogisticRegression logit;
  SVM svm;
  SVR svr;
  KNN knn;
  NaiveBayes nb;
  GaussianNaiveBayes gnb;
  std::string dataset = "hardware";
  std::string mode = "regression";

  svr.C = 1e6;
  //knn.K = 21;
  cross_validate(lin, dataset, 10, mode);
  cross_validate(svr, dataset, 10, mode);
  cross_validate_KNN(knn, dataset, 10, mode);
  test(lin, dataset, mode);
  test(svr, dataset, mode);
  test(knn, dataset, mode);


  return 0;
}
