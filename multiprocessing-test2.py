import numpy as np
from multiprocessing import Pool
import time

def matrix_multiply(A, B):
    return A @ B

def split_matrix(A, B, num_splits):
    row_size = A.shape[0] // num_splits
    col_size = B.shape[1] // num_splits

    row_splits = [A[i*row_size:(i+1)*row_size] for i in range(num_splits)]
    col_splits = [B[:, i*col_size:(i+1)*col_size] for i in range(num_splits)]

    return row_splits, col_splits

def parallel_matrix_multiply(A, B, num_splits):
    pool = Pool(processes=num_splits)
    row_splits, col_splits = split_matrix(A, B, num_splits)
    tasks = [(row_splits[i], col_splits[j]) for i in range(num_splits) for j in range(num_splits)]
    
    results = pool.starmap(matrix_multiply, tasks)

    pool.close()
    pool.join()

    final_result = np.vstack([np.hstack(results[i*num_splits:(i+1)*num_splits]) for i in range(num_splits)])
    return final_result

if __name__ == "__main__":
    n = 10000
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    num_splits = 1

    starttime = time.time()
    result = parallel_matrix_multiply(A, B, num_splits)
    print("Time taken:", time.time() - starttime)

    print(result.shape)

#ok
