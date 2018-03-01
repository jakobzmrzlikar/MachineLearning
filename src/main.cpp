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
  std::string dataset = "random_dataset";
  std::string mode = "regression";

  //cross_validate(lin, dataset, 10, mode);
  // svr.C = 1e10;
  // for (int i=0; i<10; i++) {
  //   cross_validate(svr, dataset, 20, mode);
  //   svr.C /= 10;
  // }
  svr.C = 16;
  cross_validate(svr, dataset, 10, mode);
  //cross_validate_KNN(knn, dataset, 0, mode);
  //knn.K = 21;
  //test(knn, dataset, mode);
  test(svr, dataset, mode);
  test(lin, dataset, mode);


  return 0;
}
