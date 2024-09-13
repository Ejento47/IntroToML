import numpy as np

x = np.array([[1, 2],[3,5]])
y = np.array([[2],[4]])

# w@x.T = y.T, find w
w = y.T@np.linalg.inv(x.T)
print(w)