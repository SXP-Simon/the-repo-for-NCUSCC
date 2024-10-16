# 基础实现multiprocessing库的矩阵计算
# 终端：pip install numpy
# 由于multiprocessing库为python内置，所以无须安装
import numpy as np
from multiprocessing import Pool
import time

def matrix_multiply(args):
    A, B = args
    return A @ B

def split_matrix(A, B, num_splits):
    split_size = A.shape[0] // num_splits
    splits_A = np.array_split(A, num_splits, axis=0)
    return splits_A, B

def parallel_matrix_multiply(A, B, num_splits=None):
    if num_splits is None:
        num_splits = 1
    splits_A, B = split_matrix(A, B, num_splits)
    with Pool(processes=num_splits) as pool:
        results = pool.map(matrix_multiply, [(A_block, B) for A_block in splits_A])
    result = np.vstack(results)
    return result

if __name__ == '__main__':
    n = 10000
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    num_splits = 1
    start_time = time.time()
    result = parallel_matrix_multiply(A, B, num_splits)
    print("Time taken:", time.time() - start_time)
    print(result.shape)
    np.testing.assert_allclose(result, np.dot(A, B))

#ok