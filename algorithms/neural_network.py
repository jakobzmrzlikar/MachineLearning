import random
import time
import sys
import numpy as np
from tqdm import tqdm

from functions.cost import QuadraticCost, CrossEntropyCost
from functions.functions import sigmoid, sigmoid_prime, softmax
from functions.parameters import epochs, mini_batch_size, eta, lmbda, eval_treshold, architecture

class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        self.cost = cost
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()

    def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
    def feedforward(self, x):
        i=0
        for w, b in zip(self.weights, self.biases):
            #if i != self.num_layers-2:
            x = sigmoid(np.dot(w, x) + b)
            #else:
            #    x = softmax(np.dot(w, x) + b)
            i+=1
        return x

    def SGD(self, training_data,
            evaluation_data=None,
            test_data = None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False,
            early_stopping_n = 0):

        training_data = list(training_data)
        self.n = len(training_data)

        if test_data != None:
            test_data = list(test_data)
            n_test = len(test_data)

        if evaluation_data != None:
            evaluation_data = list(evaluation_data)
            n_data = len(evaluation_data)

        for i in range(epochs):
            random.shuffle(training_data)
            for k in tqdm(range(0, self.n, mini_batch_size)):
                self.update_mini_batch(training_data[k:k+mini_batch_size])
            if test_data:
                print("Epoch {} : {} / {}".format(i, self.evaluate(test_data),n_test));
            else:
                print("Epoch {} complete".format(i))

        # Advanced monitoring functions
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []

        if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
        if monitor_training_accuracy:
            accuracy = self.accuracy(training_data, convert=True)
            training_accuracy.append(accuracy)
            print("Accuracy on training data: {} / {}".format(accuracy, n))
        if monitor_evaluation_cost:
            cost = self.total_cost(evaluation_data, lmbda, convert=True)
            evaluation_cost.append(cost)
            print("Cost on evaluation data: {}".format(cost))
        if monitor_evaluation_accuracy:
            accuracy = self.accuracy(evaluation_data)
            evaluation_accuracy.append(accuracy)
            print("Accuracy on evaluation data: {} / {}".format(self.accuracy(evaluation_data), n_data))

        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            # for each input in mini_batch, backprop calculates it's gradient
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            # then it adds it's gradients to a bach gradient so it can later
            # calculate the average gradient
            nabla_w = [i+j for i, j in zip(nabla_w, delta_nabla_w)]
            nabla_b = [i+j for i, j in zip(nabla_b, delta_nabla_b)]
        self.weights = [(1-eta*lmbda/self.n)*i-(eta/len(mini_batch)*j) for i,j in zip(self.weights, nabla_w)]
        self.biases = [i-(eta/len(mini_batch)*j) for i,j in zip(self.biases, nabla_b)]

    def backprop(self, x,y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        z_list = []
        a_list = [x]
        a = a_list[0]
        i=0
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            # for i in z:
            #     if i > 5:
            #         print(i)
            #         print()
            #         for j in b:
            #             if j > 0.5: print(j)
            #         print()
            #         for j in w:
            #             for k in j:
            #                 if k > 0.1: print(k)
            #         sys.exit(0)
            z_list.append(z)
            #if i != self.num_layers-2:
            a = sigmoid(z)
            #else:
            #    a = softmax(z)
            i+=1
            a_list.append(a)

        # calculates error for the final layer
        delta = self.cost.delta(z_list[-1], a_list[-1], y)
        nabla_w[-1] = np.dot(delta, a_list[-2].transpose())
        nabla_b[-1] = delta

        # calculates error for all other layers
        for l in range(2, self.num_layers):
            # i have no idea if this is right

            delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_prime(z_list[-l])
            nabla_w[-l] = np.dot(delta, a_list[-l-1].transpose())
            nabla_b[-l] = delta
        #print('returning nablas')
        return(nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(self.feedforward(x), y) for (x, y) in test_data]
        return sum(int(round(x[0][0])==y) for (x, y) in test_results)

    def total_cost(self, data, lmbda):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            cost += self.cost.fn(a, y)/len(data)
            cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights) # '**' - to the power of.
        return cost

if __name__  == '__main__':
    train_data = np.load('../data/train_data.npy')
    test_data = np.load('../data/test_data.npy')
    net = Network(architecture)
    net.SGD(train_data, test_data=test_data)
