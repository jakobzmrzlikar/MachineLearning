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
