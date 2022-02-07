import numpy as np

x = np.array([[1, 2, 3], [-1, -3, -4]])
y = np.array([[0], [0]])
new_X = np.append(x, y, axis = 1)
sorted_X = np.array(new_X[np.argsort(new_X[:,0])])
sorted_Y = np.array(sorted_X[:, -1])
np.delete(sorted_X, -1, axis = 1)
print(sorted_X)
print(sorted_Y)