#!/usr/bin/env python3
# 更新的第1部分算法 Segmentation and screening （在再算之前，清空之前的计算文件）
# 导入pandas包并重命名为pd
import pandas as pd
import numpy as np
import math
import copy
from scipy.stats import norm

pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=300000)
# 读取Excel中Sheet1中的数据
data = pd.DataFrame(pd.read_excel('Bolt1_1#_.xlsx', 'Sheet1'))
# data = data.drop(['支架编号'],axis=1) #删除列
# print(data.index) # 行索引
n, m = data.shape
print(data.shape)   # 获取行数和列数
print(n)      # 裁剪数据之后，为912  行数
S = 38        # the number of segments to be formed  38  T段数
L = int(n/S)  # 20 Floors,每段的长度  912/38 = 24
print("每段的长度为：", L)   #  24
T = 38  # X序列所分段数
das = n+1    # 后续好像没有用到，返回一个新的S向量

dai = []
for i in range(2, S+2):
    # range()函数包括前面，不包括后面，默认间隔为1;
    # 论文里，i 范围是 2 to S-1 ，自己认为 2 to S+1, 才能返回整个范围的时间序列值
    dai.append(L*(i-1))
print("每段结束节点为：", dai)  # 获取的cut-point（截断点） 的位置、索引值

Z = pd.DataFrame()  # 建立一个空DataFrame，获取整个时间序列
for i in dai:
    dai = data.iloc[i - L: i, 0:5]   # iloc位置索引器，行，列
    print(dai)
    Z = Z._append(dai, ignore_index=True)
Z.to_excel('Z.xlsx', columns=None, index=True)  # 即为输出整个表格

S_mean = []  # 建立一个空列表
for i in range(1, S+1):
    # range()函数包括前面，不包括后面，默认间隔为1，论文中是1 to S-2
    # 读取Excel每L段的行数以及第5列的值
    Ui = data.iloc[L*(i-1): L*(i+1), 4]  # 每一段的取值，iloc切片不包括结束端点，论文的范围：L*(i-1): L*(i+1)-1
    print("相邻两段数据为：", Ui)
    a = np.mean(Ui)   # 计算均值
    a = ('%.4f' % a)  # 均值保留4位有效数字
    S_mean.append(a)
S_mean = list(map(float, S_mean))   # 使用内置map返回一个map对象，再用list将其转换为列表
print("各段均值为：", S_mean)

# 计算 threshold 喇嘛塔 lamda
alpha = 0.01   # 设置alpha，显著性水平   个人认为其可根据数据变化调整 0.1 0.05 0.01 依次获取分割筛选段数逐渐减少
'''
a, b = norm.interval(1-alpha)
print('双侧左、右分位点：a=%.4f,b=%.4f'%(a, b))
'''
b = norm.isf(q=1-alpha)
# 计算单侧右分位点，即为上1-alpha分为点；
# 论文里标准正态分布上1-alpha分位点，则为负数，不符合论文里面的
a = -b                                            # 单侧左分位点
print('单侧左、右分位点：a=%.4f,b=%.4f' % (a, b)) 	  # 输出精确到万分位


Zi_std_ = 0
for i in range(1, S+1):
    Zi = data.iloc[L*(i-1): L*i, 4]
    print("各段值为：", Zi)
    Zi_std = np.std(Zi, ddof=1)  # 样品标准差
    Zi_std_ = Zi_std + Zi_std_   # 求和
print("样品标准差之和为:", Zi_std_)
sigma = Zi_std_/S
print("sigma的取值为：", sigma)
lamda = (math.sqrt(S)) * a * sigma/(math.sqrt(8*T))
print("临界值lamda的值为：",  lamda)

# Di = np.diff(S_mean) # 本来准备计算均值差
Di = []   # 建立一个空列表，for i in range(1, S-3):
for i in range(len(S_mean)-1):  # range()函数包括前面，不包括后面，默认间隔为1
    Di.append(abs(S_mean[i + 1] - S_mean[i]))
    # Di.append(abs(S_mean[i+1] - S_mean[i]))
Di = [float(x) for x in Di]  # 原始数据
print("相邻两段均值差为：", Di)

k = 0    # 后面好像可以用到
C = []
Ck = []

I = [] #索引列表
Di_index = []
for i in range(len(Di)):
    if Di[i] in Di[:i]:
        if Di[i] > lamda:
            I.append(I[-1] + 1 + 1 + 1)

    else:
        if Di[i] > lamda:
            I.append(Di.index(Di[i]) + 1 + 1)
print("大于临界值的段数序号为：", I)

for i in I:
    Ck = L * i  # Ck = da(i+1), da(i) = L*(i -1)，临界位置序号
    C.append(Ck)
    k = k + 1
print("临界位置个数共：", k)   # 个数
print("临界位置截止序号：", C)
for i in C:
    int(i)     # 转换数据类型，整数
    for j in range(0, L):
        print(i-L+j+1)   # 输出C所有段序列的序号、索引，方便后续处理数据
C_ = pd.DataFrame()
for i in C:
    C = data.iloc[i - L: i, 0:5]  # 获取整个C的时间序列表格
    # print(C)
    C_ = C_._append(C, ignore_index=True)
C_.to_excel('C.xlsx', columns=None, index=True)