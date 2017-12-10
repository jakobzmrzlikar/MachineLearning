import matplotlib.pyplot as plt
from algorithms.linear_regression import LinearRegression

"""
An example of linear regression in action.
"""

if __name__ == "__main__":
    h = LinearRegression()

    training_data = [(2005, 12), (2006, 19), (2007, 29), (2008, 37), (2009, 45)]
    cost = h.train(training_data)

    n = training_data[0][0]
    k = training_data[-1][0]+5
    avg = sum(i for i, _ in training_data)/len(training_data)

    points = [(x, y) for x,y in training_data]

    f = plt.figure("Hypothesis")
    plt.xlabel("x")
    plt.ylabel("h(x)")
    plt.scatter(*zip(*points), color="red")
    #input must be normalized for function to work properly
    plt.plot(range(n, k+1), [h(i-avg) for i in range(n, k+1)])
    plt.axis([n-1, k+1, 0, 100])

    g = plt.figure("Cost")
    plt.xlabel("epochs")
    plt.plot(cost)

    plt.show()
