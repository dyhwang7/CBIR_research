import numpy as np
import matplotlib.pyplot as plt
arr = [1, -5, 75, 20]

np_arr = np.array(arr)

a = np.hstack(np_arr)

_ = plt.hist(a, bins=[1,2,3,4])
plt.show()