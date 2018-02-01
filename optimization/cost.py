import matplotlib.pyplot as plt
import numpy as np

import matplotlib.cbook as cbook

data = np.genfromtxt('../data/cost.csv', delimiter='\n', skip_header=0,
                     skip_footer=0)

plt.plot(data)

plt.show()
