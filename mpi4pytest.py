from mpi4py import MPI
import numpy as np
import time

def matrix_multiply(A, B):
    """执行矩阵乘法。"""
    return np.dot(A, B)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()  # 获取总进程数

    n = 10000  # 定义矩阵大小
    # 确保矩阵可以被进程数整除
    assert n % size == 0, "Matrix size must be divisible by the number of processes."

    # 分块大小
    block_size = n // size

    if rank == 0:
        # 主进程初始化矩阵A和B
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        # 初始化用于收集结果的数组
        C = np.empty((n, n), dtype=np.float64)
    else:
        # 其他进程只需要分配空间给局部矩阵
        A_local = np.empty((block_size, n), dtype=np.float64)
        B_local = np.empty((n, block_size), dtype=np.float64)
        # 初始化用于收集结果的数组
        C_local = np.empty((block_size, block_size), dtype=np.float64)

    # 主进程分发矩阵A和B到所有进程
    if rank == 0:
        for i in range(1, size):
            # 确保发送的数组是连续的
            A_slice = A[i*block_size:(i+1)*block_size, :].copy()
            B_slice = B[:, i*block_size:(i+1)*block_size].copy()
            comm.Send(A_slice, dest=i, tag=11)
            comm.Send(B_slice, dest=i, tag=12)
        # 主进程保留第一块，避免重复计算
        A_local = A[0:block_size, :]
        B_local = B[:, 0:block_size]
        C_local = np.empty((block_size, block_size), dtype=np.float64)
    else:
        # 其他进程接收矩阵A和B的局部部分
        A_local = np.empty((block_size, n), dtype=np.float64)
        B_local = np.empty((n, block_size), dtype=np.float64)
        comm.Recv(A_local, source=0, tag=11)
        comm.Recv(B_local, source=0, tag=12)

    # 所有进程计算其分配到的矩阵片段的乘法
    start_time = time.perf_counter()
    C_local = matrix_multiply(A_local, B_local)
    end_time = time.perf_counter()

    # 计算执行时间
    execution_time = end_time - start_time

    # 主进程收集所有进程的局部结果和执行时间
    if rank == 0:
        total_time = np.empty(size, dtype=np.float64)
        for i in range(size):
            if i > 0:
                # 收集执行时间
                temp_C_local = np.empty((block_size, block_size), dtype=np.float64)
                comm.Recv(temp_C_local, source=i, tag=i)
                C[i * block_size:(i + 1) * block_size, i * block_size:(i + 1) * block_size] = temp_C_local
            else:
                total_time[0] = execution_time
                C[0:block_size, 0:block_size] = C_local
            average_time = np.mean(total_time)
            print(f"Average execution time: {average_time} seconds")
        else:
            # 其他进程发送执行时间到主进程
            comm.Send(np.array([execution_time], dtype=np.float64), dest=0, tag=rank)
            # 发送局部计算结果到主进程
            comm.Send(C_local, dest=0, tag=rank)

    if __name__ == '__main__':
        main()