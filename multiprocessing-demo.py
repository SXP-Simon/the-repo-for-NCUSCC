import numpy as np
from multiprocessing import Pool
import time


def matrix_multiply(args):
    A, B = args
    return np.dot(A, B)


def split_matrix(A, B, num_splits):
    N = A.shape[0]
    assert N % num_splits == 0, "Matrix size must be divisible by num_splits"

    split_size = N // num_splits

    splits = []
    for i in range(num_splits):
        for j in range(num_splits):
            split_A = A[i * split_size:(i + 1) * split_size, j * split_size:(j + 1) * split_size]
            split_B = B[i * split_size:(i + 1) * split_size, j * split_size:(j + 1) * split_size]
            splits.append((split_A, split_B))
    return splits


def parallel_matrix_multiply(A, B, num_splits):
    splits = split_matrix(A, B, num_splits)
    with Pool(processes=num_splits) as pool:
        results = pool.map(matrix_multiply, splits)
    # 假设每个小矩阵乘积的大小与A和B相同
    result = np.vstack([np.hstack(results[i * num_splits:(i + 1) * num_splits]) for i in range(num_splits)])
    return result


if __name__ == '__main__':
    N = 10000 # 假设矩阵大小为10000x10000
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)
    num_splits = 4

    start_time = time.time()
    result = parallel_matrix_multiply(A, B, num_splits)
    print("Time taken:", time.time() - start_time)

#ok