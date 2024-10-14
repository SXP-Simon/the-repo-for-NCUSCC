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
    print(size)

    n = 10000  # 定义矩阵大小
    block_size = n // size  # 每个进程处理的行/列数

    # 主进程初始化矩阵A和B
    if rank == 0:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
    else:
        A = None
        B = None

    # 每个进程都要有B的完整副本
    B = comm.bcast(B, root=0)

    # 为每个进程分配一部分A矩阵
    A_local = np.empty((block_size, n), dtype=np.float64)
    comm.Scatter(A, A_local, root=0)

    # 所有进程计算其分配到的矩阵片段的乘法
    start_time = time.perf_counter()
    C_local = matrix_multiply(A_local, B)
    end_time = time.perf_counter()

    # 计算执行时间
    execution_time = end_time - start_time

    # 主进程收集所有进程的执行时间并计算平均值
    if rank == 0:
        total_time = np.zeros(size, dtype=np.float64)  # 创建一个数组来存储所有进程的执行时间
        total_time[0] = execution_time
        requests = []
        for i in range(1, size):
            req = comm.Irecv(total_time[i:i + 1], source=i, tag=i)
            requests.append(req)
        # 等待所有进程发送的执行时间
        MPI.Request.Waitall(requests)
        average_time = np.mean(total_time)
        print(f"Average execution time: {average_time} seconds")
    else:
        # 其他进程发送执行时间到主进程
        comm.Isend(np.array([execution_time], dtype=np.float64), dest=0, tag=rank)

    # 主进程收集所有的C局部结果并输出
    if rank == 0:
        C = np.empty((n, n), dtype=np.float64)
    else:
        C = None

    # 使用非阻塞通信收集矩阵C
    req = comm.Igather(C_local, C, root=0)
    req.Wait()

    if rank == 0:
        print(f"Resulting matrix C: {C}")


if __name__ == '__main__':
    main()