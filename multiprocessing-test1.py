import numpy as np
from multiprocessing import Pool
import time

def matrix_multiply(A, B):
    return A @ B

def split_matrix(A, B, num_splits):
    # 计算每个子块的大小
    split_size_A = A.shape[0] // num_splits
    split_size_B = B.shape[1] // num_splits
    # 分割矩阵 A 和 B
    A_splits = [np.ascontiguousarray(A[i*split_size_A:(i+1)*split_size_A]) for i in range(num_splits)]
    B_splits = [np.ascontiguousarray(B[:, i*split_size_B:(i+1)*split_size_B]) for i in range(num_splits)]
    return A_splits, B_splits

def parallel_matrix_multiply(A, B, num_splits):
    pool = Pool(processes=num_splits)
    A_splits, B_splits = split_matrix(A, B, num_splits)
    results = []

    for i in range(num_splits):
        for j in range(num_splits):
            # 并行计算矩阵块的乘法
            result = pool.apply_async(matrix_multiply, (A_splits[i], B_splits[j]))
            results.append((i, j, result))

    pool.close()
    pool.join()

    # 初始化结果矩阵
    final_result = np.zeros((A.shape[0], B.shape[1]))

    # 将子块结果合并到最终结果矩阵中
    split_size_A = A.shape[0] // num_splits
    split_size_B = B.shape[1] // num_splits
    for i, j, result in results:
        final_result[i*split_size_A:(i+1)*split_size_A, j*split_size_B:(j+1)*split_size_B] = result.get()

    return final_result

if __name__ == "__main__":
    n = 10000
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    num_splits = 2

    starttime = time.time()
    result = parallel_matrix_multiply(A, B, num_splits)
    print("Time taken:", time.time() - starttime)

    print(result.shape)
    np.testing.assert_allclose(result, np.dot(A, B))


#ok