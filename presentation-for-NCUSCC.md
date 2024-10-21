#  **南昌大学超算俱乐部考核题（ Python ）的实验报告**
***
    FROM 回归天空
    QQ: 903928770
    微信：night-helianthus
***
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
我选择采取numba的njit装饰器对python语法进行c类语言转译加速，将尽量进行控制变量比较。
分析部分不会将比较两者的柱状图呈现，只比较不同规模的进程数效果。多方案比较将在分析部分后呈现。**

***

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
**multiprocessing结果**（出了点小问题但是没有修改，下面图的横轴为进程数，纵轴为时间（s），
依次为1000，2000，5000和10000规模的矩阵乘法）

![mp_1000.png](https://www.helloimg.com/i/2024/10/21/6715d2b5ec2c1.png)
![mp_2000.png](https://www.helloimg.com/i/2024/10/21/6715d2b618518.png)
![mp_5000.png](https://www.helloimg.com/i/2024/10/21/6715d2b5d7378.png)
![mp_10000.png](https://www.helloimg.com/i/2024/10/21/6715d2b5e6995.png)
 
    经分析：
    在数据规模较小时，并行引起的资源开销占主导地位，进程越多，时间越长，并行效率越低。
    在数据规模较大时，并行运算带来的加速效果明显，总体上进程越多，时间缩短，效率提高。
    
***

### 2.mpi4py库分析

mpi4py是一个在python中实现MPI标准的库，提供面向对象接口使得python程序可以利用多处理器进行并行运算，
但是比较适合联机玩家（多机跨节点，服务器层面），单机玩家使用时优势不明显。我在实验过程中使用了进程间集体通信和
非阻塞通信等方法进行了测试。

```python
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

```

**mpi4py结果**（由于10000规模时mpi方法时间太长，不方便计量，所以没有成功收集，
10进程下mpi方法时间超过了1800s,2进程下超过了5400s,效果较差。）

![mpi_1000.png](https://www.helloimg.com/i/2024/10/21/6715d5fa49656.png)
![mpi_2000.png](https://www.helloimg.com/i/2024/10/21/6715d5fa3c878.png)
![mpi_5000.png](https://www.helloimg.com/i/2024/10/21/6715d5fa3e936.png)

    经分析：
    在数据规模较小时，并行引起的资源开销占主导地位，进程越多，时间越长，并行效率越低。
    在数据规模较大时，并行运算带来的加速效果明显，总体上进程越多，时间缩短，效率提高。

***

#### 拓展一个多进程库：joblib

joblib是一个基于multiprocessing库的并行实现，特别优化了在数组处理和磁盘缓存发方面，
语法简洁，容易上手。在服务启动后，第一次运行可能会比后续运行多耗时一些，因为需要分配进程。
但一旦初始化完成，后续的并行计算将更加高效。

安装joblib
```terminal
pip install joblib
```
joblib并行框架例子
```python
import numpy as np
from joblib import Parallel, delayed
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
    A_splits, B_splits = split_matrix(A, B, num_splits)

    # 使用 joblib 进行并行计算
    results = Parallel(n_jobs=num_splits)(
        delayed(matrix_multiply)(A_splits[i], B_splits[j])
        for i in range(num_splits)
        for j in range(num_splits)
    )

    # 初始化结果矩阵
    final_result = np.zeros((A.shape[0], B.shape[1]))

    # 将子块结果合并到最终结果矩阵中
    split_size_A = A.shape[0] // num_splits
    split_size_B = B.shape[1] // num_splits
    for idx, (i, j) in enumerate([(i, j) for i in range(num_splits) for j in range(num_splits)]):
        final_result[i*split_size_A:(i+1)*split_size_A, j*split_size_B:(j+1)*split_size_B] = results[idx]

    return final_result

if __name__ == "__main__":
    n = 2000
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    num_splits = 8

    starttime = time.time()
    result = parallel_matrix_multiply(A, B, num_splits)
    print("Time taken for parallel matrix multiply with joblib:", time.time() - starttime)

    starttime = time.time()
    np.dot(A, B)
    print("Time taken for numpy dot product:", time.time() - starttime)

    print(result.shape)
    np.testing.assert_allclose(result, np.dot(A, B))
```
**joblib结果**（这里不做详细介绍，分别为1000，2000规模的矩阵乘法）

![joblib_1000.png](https://www.helloimg.com/i/2024/10/21/6715d926a2150.png)
![joblib_2000.png](https://www.helloimg.com/i/2024/10/21/6715d926a4bbe.png)

    joblib数据量小时无优势，数据量大时优势明显，而且在多进程资源优化方面优势明显，
    后文将用图像直观呈现。
***

### 3.多方案的比较
( 赛 博 斗 蛐 蛐 环 节 )

![1000scale.png](https://www.helloimg.com/i/2024/10/21/6715db8506d11.png)
![2000scale.png](https://www.helloimg.com/i/2024/10/21/6715db8513d37.png)
![5000scale.png](https://www.helloimg.com/i/2024/10/21/6715db851fbcf.png)
![10000scale.png](https://www.helloimg.com/i/2024/10/21/6715db8500419.png)

    分析：
    数据规模小时，mpi进程间非阻塞通讯提高了效率，良好利用了并行资源，速度较快；
    数据规模大时，mpi方法由于单机局限，造成根进程瓶颈，主进程负担过大，并行效率下降明显。
    multiprocessing库在数据规模大时效率明显高于mpi4py库，但是joblib还是比
    单纯的multiprocessing高效。但是这并不表明mpi在大数据量时计算完全比不了
    multiprocessing。
    
    网上查询结果获取的结论如下：

    “如果你的应用主要在单机上运行，且不需要跨节点的分布式计算，
    multiprocessing 可能是一个更简单、更易上手的选择。
    如果你的应用需要在多节点的集群上运行（如服务器层面），
    或者需要更高效的进程间通信和更好的可扩展性，mpi4py 可能是更好的选择。”
***

## 三.实验过程中遇到的问题

### I.优化方案优化了速度，却没有优化我的效率
使用numpy.dot()方法计算矩阵乘法确实很快，不要任何处理，可以达到10000*10000规模的矩阵乘法5秒内结束，
SciPy库里的dgemm方法也是差不多，但是如果我想参考它们并行的效率，那么所有多进程方法将跑不过单进程。最后，
我选择最原始的办法，时间复杂度为O(n^3)，但是由于时间过长，采用numba的njit装饰器加速。

```python
#numpy内置方法完成矩阵乘法
def matrix_multiply(A, B):
    return np.dot(A, B)
```

```python
#scipy中的矩阵乘法方法
import numpy as np
from scipy.linalg.blas import dgemm
import time

# 创建两个随机矩阵
n = 10000
A = np.random.rand(n, n)
B = np.random.rand(n, n)
start_time = time.time()
# 使用 dgemm 进行矩阵乘法
result = dgemm(alpha=1.0, a=A, b=B)

# 记录结束时间
end_time = time.time()

# 打印计算部分的时间
print(f"矩阵乘法计算时间：{end_time - start_time}秒")

#验证结果
np.testing.assert_allclose(result, np.dot(A, B))
```

### 2.OpenMPI安装困难
采用一堆方法安装最后没有成功，报错一直是'no module named mpi4py'，最后换用简单易用的MPICH解决。

### 3.个人方面

我在实验过程有几次从头开始的经历，最后我选择这种最复杂的方法来完成实验，几次重开有以下原因：

#### 1.Python考题选择的人数少，加上对没有系统学习过线性代数的人来说，初期比较迷茫和吃力
初期的我只能拿着题目无脑喂给AI让AI架构代码，但是弊端很大，例如错误百出但是不知道怎么修改。
我当初是由于暑假时对爬虫有相关学习，了解过一点点的multiprocessing库，才最终选择Python考题。
但是实际上手这个问题才发现问题的难点不只有并行的简单实现，我还要思考如何去选择比较优的办法实现并行，
怎么让我一个没有过多基础的人可以简单地理解代码，莫名其妙导入了一堆的库我应该怎么用它们......
由于可以交流的人很少，经常有疑惑而无处询问也没有人解答。
#### 2.早期学习氛围导致对项目比较赶，对代码逻辑方面和效率方面的理解有点打折扣
考核初期精力比较少，看到别人速通考核有点被压力到了。囫囵吞枣不求甚解地通过AI了解了一堆黑科技，
但是这些黑科技由于其本身技术成熟，经过自己的魔改滥用以后，代码提升效率不佳，微乎其微的“进步”让我的进度停滞不前，
浪费大量时间而做无用功， 最严重的是由于使用numpy库的dot 方法导致并行效率不进反退而导致自我怀疑，
考核中的大部分时间都被我拿去优化numpy.dot方法的并行计算了，但是这个方法对并行效果的呈现不会很好。
在考核时间限制最后这几天里， 由于早期对各种库知识的积累，我重新开始了考核项目的测试。
#### 3.没有系统的学习知识，资料比较难找
感觉只是看完知乎上一些科普性质的文章，对某些库的理解也不一定有很深，在考核结束后我还是将继续进行对各种
第三方库知识的深入了解，如os库,matplotlib数据可视化,pandas库等，夯实Python基础。
#### 4.rustlings的精力分散
两边着手，但是rust知识的学习时间很短，现在也忘了很多。 考核初期我是利用AI完成rustlings后
才开始对进行矩阵乘法项目的了解。这种强度对只有几个月学习经历的新手来说有点不太友好。
但是马上弃坑会让之前的rust学习白费，考核结束后会考虑重新看看rust。

***
## 特别鸣谢

***太阳王子THINKER-ONLY[https://github.com/THINKER-ONLY](https://github.com/THINKER-ONLY)***    
***浩神Howxu[https://github.com/HowXu](https://github.com/HowXu)***   
***orchid[https://github.com/orchiddell0](https://github.com/orchiddell0)***    
***龙抬头***   
***客服小祥***  
