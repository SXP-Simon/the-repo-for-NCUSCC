import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# 读取CSV文件
def read_csv(file_path):
    return pd.read_csv(file_path)


# 绘制分组柱状图
# 绘制分组柱状图
def plot_grouped_bar_chart(dataframe, process_column, time_column, script_column):
    # 设置图形的大小
    plt.figure(figsize=(12, 6))

    # 为每个script分配不同的颜色
    colors = ['blue', 'green', 'red']

    # 为每个script绘制柱状图
    width = 0.25  # 柱子的宽度
    positions = range(len(dataframe[process_column].unique()))  # 柱子的位置

    for i, script in enumerate(dataframe[script_column].unique()):
        script_data = dataframe[dataframe[script_column] == script]
        plt.bar([p + width * i for p in positions], script_data[time_column], width=width, label=script,
                color=colors[i], alpha=0.7)

    plt.title('Process Number by Time for Different Scripts')  # 图形的标题
    plt.xlabel('Process Number')  # x轴的标签
    plt.ylabel('Time')  # y轴的标签
    plt.xticks([p + width for p in positions], labels=[p for p in dataframe[process_column].unique()])  # 设置x轴的刻度和标签
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格
    plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域
    plt.show()  # 显示图形

# 主函数
def main():
    # 假设CSV文件路径为'data.csv'，且有三列数据'time', 'process_num', 'script'
    file_path = 'data.csv'
    process_column = 'process_num'  # CSV文件中进程数量数据的列名
    time_column = 'time'  # CSV文件中时间数据的列名
    script_column = 'script'  # CSV文件中脚本数据的列名

    # 读取CSV文件
    data = read_csv(file_path)

    # 绘制分组柱状图
    plot_grouped_bar_chart(data, process_column, time_column, script_column)


# 运行主函数
if __name__ == '__main__':
    main()