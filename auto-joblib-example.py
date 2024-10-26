import numpy as np
from joblib import Parallel, delayed
import time
from numba import njit
import multiprocessing
import os
import psutil
import pandas as pd
import matplotlib.pyplot as plt
import csv

# 使用 Numba 加速矩阵乘法
@njit()
def matrix_multiply(A, B):
    return np.dot(A, B)

# 分割矩阵
def split_matrix(A, B, num_splits):
    split_size = A.shape[0] // num_splits
    A_splits = [A[i*split_size:(i+1)*split_size, :] for i in range(num_splits)]
    B_splits = [B[:, i*split_size:(i+1)*split_size] for i in range(num_splits)]
    return A_splits, B_splits

# 并行矩阵乘法
def parallel_matrix_multiply(A, B, num_splits):
    A_splits, B_splits = split_matrix(A, B, num_splits)
    results = Parallel(n_jobs=num_splits)(
        delayed(matrix_multiply)(A_splits[i], B_splits[j])
        for i in range(num_splits) for j in range(num_splits)
    )
    final_result = np.zeros((A.shape[0], B.shape[1]))
    for i in range(num_splits):
        for j in range(num_splits):
            final_result[i*(A.shape[0]//num_splits):(i+1)*(A.shape[0]//num_splits),
                         j*(B.shape[1]//num_splits):(j+1)*(B.shape[1]//num_splits)] += results[i*num_splits + j]
    return final_result

# 获取 CPU 使用率
def get_cpu_usage():
    cpu_usage = psutil.cpu_percent(interval=1)
    return cpu_usage

# 获取内存使用情况
def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_usage_mb = process.memory_info().rss / (1024 * 1024)
    return memory_usage_mb

# 主函数
def main(matrix_size, num_experiments, process_counts, csv_file):
    A = np.random.rand(matrix_size, matrix_size)
    B = np.random.rand(matrix_size, matrix_size)
    reference_C = np.dot(A, B)
    for num_processes in process_counts:
        for _ in range(num_experiments):
            start_time = time.time()
            C = parallel_matrix_multiply(A, B, num_processes)
            end_time = time.time()
            cpu_usage = get_cpu_usage()
            memory_usage = get_memory_usage()
            elapsed_time = end_time - start_time
            print(f"Processes: {num_processes}, Time: {elapsed_time:.6f} seconds, Memory: {memory_usage:.2f} MB, CPU Usage: {cpu_usage}%")
            assert np.allclose(C, reference_C), "Calculation result is incorrect."
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([num_processes, elapsed_time, memory_usage, cpu_usage])

if __name__ == "__main__":
    matrix_size = 10000  # 矩阵的大小
    num_experiments = 1  # 每种规模下的实验次数
    process_counts = [1, 2, 4, 8, 10]  # 使用的进程数
    csv_file = 'performance_data.csv'
    main(matrix_size, num_experiments, process_counts, csv_file)

    # 使用 pandas 读取 CSV 文件
    df = pd.read_csv(csv_file, header=None, names=['Processes', 'Time', 'Memory Usage', 'CPU Usage'])

    # 计算相同进程数的数据的平均值
    df_avg = df.groupby('Processes').mean().reset_index()

    # 使用 matplotlib 绘制图表
    plt.figure(figsize=(10, 6))
    plt.plot(df_avg['Processes'], df_avg['Time'], marker='o', linestyle='-', label='Average Time (s)')
    plt.xlabel('Number of Processes')
    plt.ylabel('Average Time (s)')
    plt.title('Average Matrix Multiplication Performance')
    plt.legend()
    plt.grid(True)
    plt.show()