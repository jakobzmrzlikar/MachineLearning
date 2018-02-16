#ifndef DATALOADER_H
#define DATALOADER_H

#include <vector>

typedef std::vector<std::vector<double>> data;

data load(std::string filename);

#endif /* DATALOADER_H */
