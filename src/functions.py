import numpy as np

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def softmax(z):
    #print(z, type(z))
    a_list = []
    summa = 0
    for k in z:
        summa += np.exp(k)
    for i in range(len(z)):
        a = np.exp(z[i])
        a_list.append(a/summa)
    # print(a_list, a_list[0], type(a_list[0]))
    # print()
    return(a_list)
