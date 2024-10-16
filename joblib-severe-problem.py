import numpy as np
from joblib import Parallel, delayed
import time


def matrix_multiply(A, B):
    return A @ B


def split_matrix(A, B, num_splits):
    # 计算每个块的尺寸
    row_splits = np.array_split(np.arange(A.shape[0]), num_splits)
    A_blocks = [np.ascontiguousarray(A[r1[0]:r1[-1], :]) for r1 in row_splits]

    # 对于矩阵 B，我们需要按照列分割
    col_splits = np.array_split(np.arange(B.shape[1]), num_splits)
    B_blocks = [np.ascontiguousarray(B[:, c1[0]:c1[-1]]) for c1 in col_splits]

    return A_blocks, B_blocks


def parallel_matrix_multiply(A, B, num_cores):
    # 将矩阵A和B分解为子块
    A_blocks, B_blocks = split_matrix(A, B, num_cores)

    # 使用delayed函数包装矩阵乘法函数
    results = Parallel(n_jobs=num_cores)(
        delayed(matrix_multiply)(A_block, B_block) for A_block, B_block in zip(A_blocks, B_blocks))

    # 将子块的结果合并为最终结果
    result = np.vstack(results)

    return result


# 示例矩阵
n = 10000
A = np.random.rand(n, n)
B = np.random.rand(n, n)

# 并行计算矩阵乘法
num_cores = 8
starttime = time.time()
result = parallel_matrix_multiply(A, B, num_cores)
print("Time taken:", time.time() - starttime)

print(result.shape)
np.testing.assert_allclose(result,np.dot(A,B))





#sereve problem
