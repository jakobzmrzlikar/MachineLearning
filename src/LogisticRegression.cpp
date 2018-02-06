#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "LogisticRegression.hpp"
#include "functions.hpp"

// training_data vector consisting of pairs (x,y)
typedef std::vector<std::vector<double>> data;

LogisticRegression::LogisticRegression(int features){
    for (int i=0; i<features; i++) {
        w.push_back(0.0);
    }
}

double LogisticRegression::h(std::vector<double> x) {
    double result = w[0];
    for (int i=1; i<w.size(); i++) {
        result+=w[i] * x[i-1];
    }
    result = Functions::sigmoid(result);
    return result;
}

double LogisticRegression::cost(data training_data, int m) {
    double cost = 0.0;
    for (int i=0; i<m; i++) {
        std::vector<double> x(training_data[i].begin(), training_data[i].end()-1);
        double y = training_data[i].back();
        double a = h(x);
        cost += CrossEntropyCost::cost(a, y);
    }

    return cost/m;
}


void LogisticRegression::save(std::string filename) {
    std::string name = "../data/" + filename;
    std::ofstream file(name);
    file << w[0];
    for (int i=1; i<w.size(); i++) {
      file << "," << w[i];
    }
    file << std::endl;
    std::cout << "Model saved to path: " << name << '\n';
}

void LogisticRegression::load(std::string filename) {
    std::string name = "../data/" + filename;
    std::ifstream file(name);
    std::string line;
    w.clear();

    while(getline(file,line)) {
        std::stringstream linestream(line);
        std::string value;

        while(getline(linestream,value,',')) {
            w.push_back(atof(value.c_str()));
        }

    }

}
