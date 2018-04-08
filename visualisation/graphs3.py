import matplotlib.pyplot as plt
import numpy as np

deltax = np.random.uniform(-1,1, size=(100,))
deltay = np.random.uniform(-1,1, size=(100,))
x = np.full((1, 100), deltax)
y = np.full((1, 100), deltay)

x2 = np.random.uniform(-2,2, size=(100,))


plt.figure("Metoda podpornih vektorjev")
plt.subplot(111)
plt.xlabel("X")
plt.ylabel("Y")
plt.scatter(x, y)
for i in range(len(x2)):
    plt.scatter(x2[i], np.sqrt(4-np.square(x2[i])), color='r')
for i in range(len(x2)):
    plt.scatter(x2[i], -np.sqrt(4-np.square(x2[i])), color='r')


plt.show()
