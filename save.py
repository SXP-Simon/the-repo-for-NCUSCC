
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt
import subprocess


def run_script(script_name):
    start_time = time.time()
    # 使用 subprocess.run 来执行脚本，并等待其完成
    exec="C:\我的domain\python学习\python项目\pythonProject\.venv\Scripts\python.exe"
    subprocess.run([exec, script_name],capture_output=True,text=True)
    end_time = time.time()
    return end_time - start_time


def save_results_to_csv(results, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Script', 'Time'])
        for result in results:
            writer.writerow(result)


def plot_results(df, output_pdf, output_svg):
    plt.figure(figsize=(10, 6))
    plt.bar(df['Script'], df['Time'], color=['blue', 'green', 'red'])
    plt.xlabel('Script')
    plt.ylabel('Time (seconds)')
    plt.title('Comparison of Parallel Matrix Multiplication Scripts')

    # 保存为 PDF 格式
    plt.savefig(output_pdf, format='pdf')

    # 保存为 SVG 格式
    plt.savefig(output_svg, format='svg')

    # 显示图表
    plt.show()


if __name__ == "__main__":
    scripts = ['multiprocessing_script.py', 'mpi_script.py', 'joblib_script.py']
    results = []

    for script in scripts:
        print(f"Running {script}...")
        elapsed_time = run_script(script)
        results.append([script, elapsed_time])
        print(f"Time taken for {script}: {elapsed_time} seconds")

    # 保存结果到 CSV 文件
    save_results_to_csv(results, 'parallel_matrix_multiply_results.csv')

    # 使用 pandas 读取 CSV 文件
    df = pd.read_csv('parallel_matrix_multiply_results.csv')

    # 使用 matplotlib 绘制图表并保存为矢量图
    plot_results(df, 'parallel_matrix_multiply_results.pdf', 'parallel_matrix_multiply_results.svg')