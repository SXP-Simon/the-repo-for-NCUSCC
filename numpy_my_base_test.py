import numpy as np
import time



n = 10000
A = np.random.rand(n, n)
B = np.random.rand(n, n)

starttime = time.time()
np.dot(A,B)
print("Time taken:", time.time() - starttime)