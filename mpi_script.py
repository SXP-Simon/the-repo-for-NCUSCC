import numpy as np
from mpi4py import MPI
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
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 计算每个进程的块索引
    block_per_process = (num_splits * num_splits) // size
    start_block = rank * block_per_process
    end_block = (rank + 1) * block_per_process

    # 分割矩阵
    A_splits, B_splits = split_matrix(A, B, num_splits)

    # 初始化结果矩阵
    final_result = np.zeros((A.shape[0], B.shape[1]))

    # 每个进程计算自己的块
    requests = []
    for block_idx in range(start_block, end_block):
        i = block_idx // num_splits
        j = block_idx % num_splits
        result_block = matrix_multiply(A_splits[i], B_splits[j])

        # 使用非阻塞发送将结果发送给根进程
        req = comm.Isend(result_block, dest=0, tag=block_idx)
        requests.append(req)

    # 根进程收集所有结果
    if rank == 0:
        split_size_A = A.shape[0] // num_splits
        split_size_B = B.shape[1] // num_splits
        for block_idx in range(num_splits * num_splits):
            i = block_idx // num_splits
            j = block_idx % num_splits
            result_block = np.zeros((split_size_A, split_size_B))
            req = comm.Irecv(result_block, source=MPI.ANY_SOURCE, tag=block_idx)
            req.Wait()
            final_result[i*split_size_A:(i+1)*split_size_A, j*split_size_B:(j+1)*split_size_B] = result_block

    # 等待所有非阻塞发送完成
    MPI.Request.Waitall(requests)

    return final_result

if __name__ == "__main__":
    n = 10000
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    num_splits = 10

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        starttime = time.time()
        result = parallel_matrix_multiply(A, B, num_splits)
        print("Time taken for parallel matrix multiply with MPI (non-blocking):", time.time() - starttime)

        starttime = time.time()
        np.dot(A, B)
        print("Time taken for numpy dot product:", time.time() - starttime)

        print(result.shape)
        np.testing.assert_allclose(result, np.dot(A, B))