import numpy as np
from joblib import Parallel, delayed
import time
from numba import njit

@njit()
def matrix_multiply(A_block, B_block):
    # 初始化结果矩阵块
    result_block = np.zeros((A_block.shape[0], B_block.shape[1]))
    # 使用基本的矩阵乘法实现
    for i in range(A_block.shape[0]):
        for j in range(B_block.shape[1]):
            for k in range(A_block.shape[1]):
                result_block[i, j] += A_block[i, k] * B_block[k, j]
    return result_block
@njit()
def split_matrix(A, B, num_splits):
    # 计算每个子块的大小
    split_size_A = A.shape[0] // num_splits
    split_size_B = B.shape[1] // num_splits
    # 分割矩阵 A 和 B
    A_splits = [np.ascontiguousarray(A[i*split_size_A:(i+1)*split_size_A]) for i in range(num_splits)]
    B_splits = [np.ascontiguousarray(B[:, i*split_size_B:(i+1)*split_size_B]) for i in range(num_splits)]
    return A_splits, B_splits

def parallel_matrix_multiply(A, B, num_splits):
    A_splits, B_splits = split_matrix(A, B, num_splits)

    # 使用 joblib 进行并行计算
    results = Parallel(n_jobs=num_splits)(
        delayed(matrix_multiply)(A_splits[i], B_splits[j])
        for i in range(num_splits)
        for j in range(num_splits)
    )

    # 初始化结果矩阵
    final_result = np.zeros((A.shape[0], B.shape[1]))

    # 将子块结果合并到最终结果矩阵中
    split_size_A = A.shape[0] // num_splits
    split_size_B = B.shape[1] // num_splits
    for idx, (i, j) in enumerate([(i, j) for i in range(num_splits) for j in range(num_splits)]):
        final_result[i*split_size_A:(i+1)*split_size_A, j*split_size_B:(j+1)*split_size_B] = results[idx]

    return final_result

if __name__ == "__main__":
    n = 2000
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    num_splits = 8

    starttime = time.time()
    result = parallel_matrix_multiply(A, B, num_splits)
    print("Time taken for parallel matrix multiply with joblib:", time.time() - starttime)

    starttime = time.time()
    np.dot(A, B)
    print("Time taken for numpy dot product:", time.time() - starttime)

    print(result.shape)
    np.testing.assert_allclose(result, np.dot(A, B))