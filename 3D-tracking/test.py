import numpy as np
def calculate_variance_loop(arr):
    if len(arr) < 2:
        raise ValueError("Array must contain at least 2 elements")
    mean = sum(arr) / len(arr)
    variance = 0
    for num in arr:
        variance += (num - mean) ** 2
    variance /= (len(arr) - 1)  # Use n-1 for unbiased variance
    return variance


h = np.var([1004.2, 1000.5, 1070])
print(h)