import numpy as np
from multiprocessing import Pool
import time

def matrix_multiply(A, B):
    return A @ B

def split_matrix(A, B, num_splits):
    row_splits = np.array_split(A, num_splits, axis=0)
    col_splits = np.array_split(B, num_splits, axis=1)
    return row_splits, col_splits

def parallel_matrix_multiply(A, B, num_splits):
    pool = Pool(processes=num_splits)
    row_splits, col_splits = split_matrix(A, B, num_splits)
    results = []

    for i in range(num_splits):
        result = pool.apply_async(matrix_multiply, (row_splits[i], col_splits[i]))
        results.append(result)

    pool.close()
    pool.join()

    final_result = np.vstack([result.get() for result in results])
    return final_result

if __name__ == "__main__":
    n = 10000
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    num_splits = 4

    starttime = time.time()
    result = parallel_matrix_multiply(A, B, num_splits)
    print("Time taken:", time.time() - starttime)