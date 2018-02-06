#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdlib>
#include "DataLoader.hpp"

typedef std::vector<std::vector<double>> data;


data DataLoader::load(std::string filename) {
  std::string name;
  std::string line;
  std::vector<double> vec;
  data training_data;
  std::string value;

  name = "../data/" + filename;
  std::ifstream file(name);

  while(getline(file,line)) {
    std::stringstream linestream(line);


    while(getline(linestream,value,',')) {
        vec.push_back(atof(value.c_str()));
    }
    training_data.push_back(vec);
    vec.clear();
  }

  return training_data;

}
