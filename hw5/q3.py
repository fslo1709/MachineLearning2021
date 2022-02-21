from osqp
import numpy as np
from scipy import sparse

# Define problem data
P = sparse.csc_matrix([[0, 0, 0], [0, 1, 0], [0, 0, 1]])
q = np.array([0, 0, 0])
A = sparse.csc_matrix([[-1, 0, -1], [-1, 0, 0], [-1, 0, 1], [1, 1, 0]])
l = np.array([1, 1, 1, 1])
u = np.array([4, 4, 4, 4])

# Create an OSQP object
prob = osqp.OSQP()

# Setup workspace and change alpha parameter
prob.setup(P, q, A, l, u, alpha = 1.0)

# Solve problem
res = prob.solve()
print(res.x)