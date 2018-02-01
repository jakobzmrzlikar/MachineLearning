import matplotlib.pyplot as plt
import numpy as np

cost = np.genfromtxt('../data/cost.csv', delimiter='\n', skip_header=0,
                     skip_footer=0)
data = np.genfromtxt('../data/train.csv', delimiter=',', skip_header=0,
                     skip_footer=0, names=['x', 'y'])
x=data['x']
y=data['y']


plt.figure("Linear regression")

plt.subplot(211)
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(x,y)


plt.subplot(212)
plt.xlabel("epochs")
plt.ylabel("cost")
plt.plot(cost)

plt.show()
