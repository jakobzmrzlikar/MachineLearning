import matplotlib.pyplot as plt
import numpy as np

deltax = np.random.normal(-1,1, size=(100,))
deltay = np.random.normal(-1,1, size=(100,))
x = np.full((1, 100), 2+deltax)
y = np.full((1, 100), 2+deltay)

deltax2 = np.random.normal(-1,1, size=(100,))
deltay2 = np.random.normal(-1,1, size=(100,))
x2 = np.full((1, 100), 8+deltax)
y2 = np.full((1, 100), 8+deltay)

plt.figure("Metoda podpornih vektorjev")
plt.subplot(111)
plt.xlabel("X")
plt.ylabel("Y")
plt.scatter(x, y)
plt.scatter(x2, y2, color='red')
plt.plot([10, 2], [-2, 10], linewidth=3, color='green', linestyle='--')
plt.plot([10, -2], [-2, 10], linewidth=3, color='purple', linestyle='-')
plt.plot([-5, 15], [4, 4], linewidth=3, color='orange', linestyle=':')

plt.show()

#plt.savefig("SVM.png")
