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
    if rank == 0:
        # 主进程初始化矩阵A和B
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        # 分发矩阵A和B到所有进程
        A_local = np.empty((n // size, n), dtype=np.float64)
        B_local = np.empty((n, n // size), dtype=np.float64)
        comm.Scatter(A, A_local, root=0)
        comm.Scatter(B, B_local, root=0)
    else:
        # 其他进程初始化接收矩阵A和B的本地部分
        A_local = np.empty((n // size, n), dtype=np.float64)
        B_local = np.empty((n, n // size), dtype=np.float64)

    # 所有进程计算其分配到的矩阵片段的乘法
    start_time = time.perf_counter()
    C_local = matrix_multiply(A_local, B_local)
    end_time = time.perf_counter()

    # 计算执行时间
    execution_time = end_time - start_time

    # 主进程收集所有进程的执行时间并计算平均值
    if rank == 0:
        total_time = np.zeros(size, dtype=np.float64)  # 创建一个数组来存储所有进程的执行时间
        # 将执行时间放入数组的第一个元素
        total_time[0] = execution_time
        # 其他进程将执行时间放入数组的对应位置
        for i in range(1, size):
            comm.Recv(total_time[i:i+1], source=i, tag=i)
        average_time = np.mean(total_time)
        print(f"Average execution time: {average_time} seconds")
    else:
        # 其他进程发送执行时间到主进程
        comm.Send(np.array([execution_time], dtype=np.float64), dest=0, tag=rank)

if __name__ == '__main__':
    main()

#ok