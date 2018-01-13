#include <fstream>
#include <vector>
#include <iterator>
#include <string>
#include <algorithm>
#include <iostream>
#include <boost/algorithm/string.hpp>

/*
 * A class to read data from a csv file.
 */
class CSVReader {
    std::string fileName;
    std::string delimeter;

public:
    CSVReader(std::string filename, std::string delm = ",") :
            fileName(filename), delimeter(delm)
    { }

    // Function to fetch data from a CSV File
    std::vector<std::vector<std::string> > getData() {

        /*
        * Parses through csv file line by line and returns the data
        * in vector of vector of strings.
        */
        std::ifstream file(fileName);

        std::vector<std::vector<std::string> > dataList;

        std::string line = "";
        // Iterate through each line and split the content using delimeter
        while (getline(file, line)) {
            std::vector<std::string> vec;
            boost::algorithm::split(vec, line, boost::is_any_of(delimeter));
            dataList.push_back(vec);
        }
        // Close the File
        file.close();

        return dataList;
    }
};

class DataLoader {
    std::string path;
    std::vector<std::vector<double>> training_data;

public:
    DataLoader(std::string filename) : path("../data/" + filename) {}

    std::vector<std::vector<double>> load() {
        // Creating an object of CSVWriter
        CSVReader reader(path);

        // Get the data from CSV File
        std::vector<std::vector<std::string> > dataList = reader.getData();

        // Convert data from string to double
        for (std::vector<std::string> vec : dataList) {
            std::vector<double> temp;
            for(std::string data : vec) {
                temp.push_back(atof(data.c_str()));
            }
            training_data.push_back(temp);
        }

        return training_data;

    }
};

