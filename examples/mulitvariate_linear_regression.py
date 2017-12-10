import matplotlib.pyplot as plt
from algorithms.mulltivariate_linear_regression import MultivariateLinearRegression

if __name__ == "__main__":

    # TODO: move data loading to preprocessing

    # load data
    training_data = []
    while True:
        line = input()
        if line in ['\n', '\r\n']:
            break
        line.split()
        line = line[1:]
        line = map(int, line)
        training_data.append(line)

    #h = MultivariateLinearRegression()
    #cost = h.train(training_data)
