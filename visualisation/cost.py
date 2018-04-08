import matplotlib.pyplot as plt
import numpy as np

cost = np.genfromtxt('../data/classification/spambase/training_cost.csv', delimiter='\n', skip_header=0,
                     skip_footer=0)
# data = np.genfromtxt('../data/train.csv', delimiter=',', skip_header=0,
#                      skip_footer=0, names=['x', 'y'])
# weights = np.genfromtxt('../data/save.csv', delimiter=',', skip_header=0,
#                      skip_footer=0)
#
# x_data=data['x']
# y_data=data['y']
#
# x = [min(x_data), max(x_data)]
# y = [weights[0]+x[0]*weights[1], weights[0]+x[1]*weights[1]]
#
#
# plt.figure("Linear regression")
#
# plt.subplot(211)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.scatter(x_data,y_data)
# plt.plot(x, y, 'r',linewidth=2)

plt.figure("SVM")
plt.subplot(212)
plt.xlabel("epochs")
plt.ylabel("cost")
plt.plot(cost,linewidth=2.5)

plt.show()
