import numpy as np
from joblib import Parallel, delayed
import time


def matrix_multiply(A, B):
    return A @ B


def split_matrix(A, num_splits):
    # 按行将矩阵 A 分块，不对 B 进行拆分
    row_splits = np.array_split(np.arange(A.shape[0]), num_splits)
    A_blocks = [np.ascontiguousarray(A[r[0]:r[-1] + 1, :]) for r in row_splits]
    return A_blocks


def parallel_matrix_multiply(A, B, num_cores):
    # 将矩阵A按行分块
    A_blocks = split_matrix(A, num_cores)

    # 使用delayed函数包装矩阵乘法函数
    results = Parallel(n_jobs=num_cores)(
        delayed(matrix_multiply)(A_block, B) for A_block in A_blocks)

    # 将结果按行堆叠
    result = np.vstack(results)

    return result


# 示例矩阵
n = 10000
A = np.random.rand(n, n)
B = np.random.rand(n, n)

# 并行计算矩阵乘法
num_cores = 2
starttime = time.time()
result = parallel_matrix_multiply(A, B, num_cores)
print("Time taken:", time.time() - starttime)

print(result.shape)
np.testing.assert_allclose(result,np.dot(A,B))

#ok

