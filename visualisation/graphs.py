import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-50, 50)
delta = np.random.uniform(-100,100, size=(100,))
#y = x + delta
y = 1/(1+np.exp(-x+delta))

y2 = 1/(1+np.exp(-x))

plt.figure("Logistična regresija")
plt.subplot(111)
plt.xlabel("X")
plt.ylabel("Y")
plt.scatter(x, y)
plt.plot(x, y2, linewidth=1, color='r')

plt.savefig("logistic_regression.png")
