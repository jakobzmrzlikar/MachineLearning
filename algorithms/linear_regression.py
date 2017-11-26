import matplotlib.pyplot as plt

class QuadraticCost():

    @staticmethod
    def cost(a, y):
        return(0.5*(a-y)**2)

    @staticmethod
    def error_b(a, y):
        return(a-y)

    @staticmethod
    def error_w(a, x, y):
        return((a-y)*x)

class MeanNormalization():
    @staticmethod
    def normalize(training_data):
        avg = sum(i for i, _ in training_data)/len(training_data)
        return([((x-avg), y) for x,y in training_data])

    @staticmethod
    def scale(training_data):
        scale = max(training_data)[0] - min(training_data)[0]
        return([((x/scale), y) for x,y in training_data])


class LinearRegression():
    def __init__(self, cost_function=QuadraticCost):
        self.b = 0
        self.w = 0
        self.cost_function = cost_function

    def __call__(self, x):
        return(self.b+self.w*x)

    def train(self, training_data, learinig_rate=0.1, epochs=1000, normalization=MeanNormalization):
        m = len(training_data)

        training_data = normalization.normalize(training_data)

        cost = [sum([self.cost_function.cost(self(x), y) for x,y in training_data]) / m]

        for i in range(epochs):
            nabla_b = sum([self.cost_function.error_b(self(x), y) for x,y in training_data])
            nabla_w = sum([self.cost_function.error_w(self(x), x, y) for x,y in training_data])

            self.b -= learinig_rate / m * nabla_b
            self.w -= learinig_rate / m * nabla_w
            cost.append(sum([self.cost_function.cost(self(x), y) for x,y in training_data]) / m)

        return cost

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
