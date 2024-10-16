#import time
import pandas as pd
import matplotlib.pyplot as plt

# 假设您已经有了两个方案的计算时间数据
# 方案1的时间数据
timescheme1 = [0.1, 0.2, 0.3, 0.4, 0.5]
# 方案2的时间数据
timescheme2 = [0.05, 0.1, 0.15, 0.2, 0.25]

# 创建一个DataFrame
data = {'Task': range(1, 6), 'Scheme1': timescheme1, 'Scheme2': timescheme2}
df = pd.DataFrame(data)

# 将数据写入CSV文件
df.to_csv('matrix_computation_times.csv', index=False)

# 使用pandas读取CSV文件
df = pd.read_csv('matrix_computation_times.csv')

# 使用matplotlib绘制矢量图
plt.figure(figsize=(10, 6))
plt.plot(df['Task'], df['Scheme1'], marker='o', label='Scheme 1')
plt.plot(df['Task'], df['Scheme2'], marker='s', label='Scheme 2')
plt.title('Matrix Computation Times')
plt.xlabel('Task')
plt.ylabel('Time (s)')
plt.legend()
plt.grid(True)

# 保存图表为文件
plt.savefig('matrix_computation_times.png')

# 显示图表
plt.show()