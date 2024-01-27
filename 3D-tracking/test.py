import numpy as np
from scipy.optimize import linear_sum_assignment

# Assuming you have a matrix A(N, M)
# Replace this with your actual matrix
A = np.array([[5, 9, 1],
              [10, 3, 2],
              [8, 7, 4]])

# Apply the Hungarian algorithm to find the assignment
row_ind, col_ind = linear_sum_assignment(A, maximize=True)

# Sort the assignments based on the values in the matrix
sorted_indices = np.argsort(-A[row_ind, col_ind])

# Get the top 5 pairs
for i, j in zip(row_ind[sorted_indices][:2], col_ind[sorted_indices][:2]):
    print(i, j)