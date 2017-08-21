"""multiple_eta
~~~~~~~~~~~~~~~
This program shows how different values for the learning rate affect
training.  In particular, we'll plot out how the cost changes using
three different values for eta.
"""

# Standard library
import json
import random
import sys

# My library
sys.path.append('../src/')
from main import Network
from constants import epochs, architecture

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np

# Constants
LEARNING_RATES = [0.025, 0.25, 2.5]
COLORS = ['#2A6EA6', '#FFCD33', '#FF7033']

def main():
    run_networks()
    make_plot()

def run_networks():
    """Train networks using three different values for the learning rate,
    and store the cost curves in the file ``multiple_eta.json``, where
    they can later be used by ``make_plot``.
    """
    # Make results more easily reproducible
    random.seed(12345678)
    np.random.seed(12345678)
    train_data = np.load('../data/train_data.npy')
    test_data = np.load('../data/test_data.npy')
    results = []
    for eta in LEARNING_RATES:
        print("\nTrain a network using eta =" + str(eta))
        net = Network(architecture)
        results.append(
            net.SGD(train_data, monitor_training_cost=True))
    f = open("multiple_eta.json", "w")
    json.dump(results, f)
    f.close()

def make_plot():
    f = open("multiple_eta.json", "r")
    results = json.load(f)
    f.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for eta, result, color in zip(LEARNING_RATES, results, COLORS):
        _, _, training_cost, _ = result
        print(training_cost, type(training_cost))
        ax.plot(np.arange(epochs), training_cost[0], "o-",
                label="$\eta$ = "+str(eta),
                color=color)
    ax.set_xlim([0, epochs])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cost')
    plt.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    main()
