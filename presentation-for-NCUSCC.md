#  **南昌大学超算俱乐部考核题（ Python ）的实验报告**

    FROM 回归天空
    QQ: 903928770
    微信：night-helianthus

## **考核要求：**
![ECCF30E1EE39A0B2A385159BBBCAF462.jpg](https://www.helloimg.com/i/2024/10/20/671484f1c731b.jpg)
![7DACA7EB17E65C88127FE184F59C2501.jpg](https://www.helloimg.com/i/2024/10/20/671484f1b0a8b.jpg)
![7DDB47BE84E236ED7E64DE2149860080.jpg](https://www.helloimg.com/i/2024/10/20/671484f1ac073.jpg)
![CC7D6C100E085087492A8D895BB8BE61.jpg](https://www.helloimg.com/i/2024/10/20/671484f182956.jpg)

## 一.环境搭建

### 1.配置虚拟机：在虚拟机中安装 Ubuntu 22.04 LTS 操作系统。

使用VMware Workstation，首先在镜像站（如清华源）中下载ubuntu镜像文件，根据安装指南完成vm安装，
创建ubuntu64位虚拟机，由于要进行多进程测试，将处理器部分的处理器数量设置为4，每个处理器内核数为4，
网络设置为最方便使用的NAT模式，安装ubuntu系统，在Software&updates中换源（如阿里云），完成基本配置。

### 2.安装IDE: pycharm

**第一种**最容易理解的办法是到类似应用商店的 ubuntu software 中下载，只需搜索+点击下载即可。首先确保更新，
然后搜索pycharm community edition，终端更新指令如下。
```bash
sudo apt update
```
**第二种**则是通过snap包管理器，终端进行安装
```bash
sudo apt update
sudo apt install snapd
sudo snap install pycharm-communiy --classic
```
完成后终端输入来启动并检查pycharm的安装
```bash
pycharm-community
```
如果想要更新pycharm,终端指令如下
```bash
sudo snap refresh pycharm-community
```

### 3.pycharm虚拟环境中必要的环境配置
**pip包管理器安装**
创建一个新项目后，打开终端，进行pip包管理器安装
```bash
sudo apt update
sudo apt install python3-pip
```

**安装必要的库**
打开项目终端，输入安装指令(这里只以numpy为例)
```terminal
pip install numpy
```
**安装MPI**
终端安装mpich
```bash
sudo apt-get update
sudo apt-get install mpich
```
检查是否安装mpich
```bash
mpicc -v
```
再安装mpi4py库
```terminal
pip install mpi4py
```

### 4.git的环境搭建
github与git的配置，登录github，配置Ubuntu的.ssh文件，将公钥上传github，方便后期git上传与拉取
终端命令生成ssh对：
```bash
ssh-keygen -t rsa -b 4096 -C "邮箱地址"
```
配置完ssh后测试与github的连接情况
```bash
ssh -T git@github.com
```
生成项目并克隆仓库到本地
```bash
cd ~/PycharmProjects/pythonProject
git clone git@github.com:SXP-Simon/the-repo-for-NCUSCC.git
```
若要实现ubuntu无障碍连接到www.github.com,推荐使用金钱方面无痛但是需要配置一点证书的watt tooltik。
至此，已经基本搭建测试所需要的环境了。

## 二.考核部分的multiprocessing与mpi4py的比较

**在此声明，我的本次实验中选择的矩阵乘法计算方法为最原始的手撕矩阵方法，使用for循环的嵌套，
时间复杂度为O(n^3),至于原因我在后文实验过程中遇到的问题部分会做出回答。由于时间复杂度过大，
我选择采取numba的njit装饰器对python语法进行c类语言转译加速，两个方法将尽量进行控制变量比较。**

### 1.multiprocessing库的分析

        multiprocessing中使用简洁的语法来调用多核编程方法，简化了实现多进程编程的复杂性，适合单机玩家享受。
    这里我选择比较有效的进程池方法与异步方法来进行并行计算，采用with方法来自动管理资源。

```python
import numpy as np
from multiprocessing import Pool
import time
from numba import njit

@njit
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
    A_splits = [A[i*split_size_A:(i+1)*split_size_A] for i in range(num_splits)]
    B_splits = [B[:, i*split_size_B:(i+1)*split_size_B] for i in range(num_splits)]
    return A_splits, B_splits

def parallel_matrix_multiply(A, B, num_splits):
    A_splits, B_splits = split_matrix(A, B, num_splits)

    # 创建进程池
    with Pool(processes=num_splits) as pool:
        # 使用 starmap 进行并行计算
        results = pool.starmap(matrix_multiply, [(A_splits[i], B_splits[j]) for i in range(num_splits) for j in range(num_splits)])

    # 初始化结果矩阵
    final_result = np.zeros((A.shape[0], B.shape[1]))

    # 将子块结果合并到最终结果矩阵中
    split_size_A = A.shape[0] // num_splits
    split_size_B = B.shape[1] // num_splits
    for idx, (i, j) in enumerate([(i, j) for i in range(num_splits) for j in range(num_splits)]):
        final_result[i*split_size_A:(i+1)*split_size_A, j*split_size_B:(j+1)*split_size_B] = results[idx]

    return final_result

if __name__ == "__main__":
    n = 10000
    #矩阵大小声明
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    num_splits = 8
    #进程数声明

    starttime = time.time()
    result = parallel_matrix_multiply(A, B, num_splits)
    print("Time taken for parallel matrix multiply with numba:", time.time() - starttime)

    print(result.shape)
    #验证矩阵形状
    np.testing.assert_allclose(result, np.dot(A, B))
    #验证计算结果
```
### 2.mpi4py库分析

        mpi4py是一个在python中实现MPI标准的库，提供面向对象接口使得python程序可以利用多处理器进行并行运算，
    但是比较适合联机玩家（多机跨节点，服务器层面），单机玩家使用时优势不明显。我在实验过程中使用了进程间集体通信和
    非阻塞通信等方法进行了测试。

```python
...


#待补全

```


