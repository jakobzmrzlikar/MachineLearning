import numpy as np

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def softmax(z):
    a_list = []
    summa = 0
    for k in z:
        summa += np.exp(k)
    for i in range(len(z)):
        a = np.exp(z[i])
        a_list.append(a/summa)
    return(a_list)
