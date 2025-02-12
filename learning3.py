#!/usr/bin/env python3
# 第3部分 Peak Recognition
# 导入pandas包并重命名为pd
import pandas as pd
import numpy as np
import math
import xlwt
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, DateFormatter
import xlrd

pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
data = pd.DataFrame(pd.read_excel('Bolt1_1#_.xlsx', 'Sheet1'))
C = pd.DataFrame(pd.read_excel('C.xlsx', 'Sheet1'))

cost_ = pd.DataFrame(pd.read_excel('2-cost_Z.xlsx', 'cost_Z'))     # 处理之后的文件
p, q = cost_.shape
cost = cost_.iloc[:, 2]  # 获取第3列数据值, cost的值
l = cost_.iloc[:, 1]     # 获取第2列数据值，lc的序号
L_ = cost_.iloc[:, 0]    # 获取第1列数据值，L_\100 = j或者k
print(cost)
print(data.index)  # 行索引
print(C.shape)     # 获取行数和列数
n, m = data.shape  # n =
a, b = C.shape
S = 38              #  the number of segments to be formed,裁剪完
L = int(n/S)       #  Floor(n\s)
print("每段的长度：", L) #48 24
lc = a/L   # 获取行数 lc = Length(C), lc = ，裁剪完不是这个值
lc = int(lc)
print("C存在的段数：", lc)

h = 6         #  a bandwidth，论文中没有提及怎么选取
gama = 0.75
cp = 1
D = []
AREA= pd.DataFrame()
# 计算局部，每一段局部遍历循环，获得局部突变点索引
for i in range(1, lc + 1):    # 分段判断，cost(i,j)的i
    print('目前段数：', i) # 一段一段局部的算
    # max, lmax, j0, Dxr = 0, Dxl =0, cp = 1，初始化各种值
    max = float(0)
    lmax = 0
    j0 = 0
    Dxr = 0
    Dxl = 0
    ci = 0
    costi = cost_.iloc[L*(i-1): L*i, 2] #文件第三列
    # print("目前段数内cost为：", costi)
    for cost in cost_.iloc[L*(i-1): L*i, 2]:  # 范围是1-24，……
        # print(cost)
        if cost > max:
            max = cost
            ci = ci + 1  # ci 计数函数
    # 结束循环，获得该段的结果
    print('当前段的计数函数为：', ci)  # 指示函数
    print('当前段最大值为：', max)  # 循环获得最大值而已
    d = list(costi).index(max)  # 寻找在该段列表的最大值位置索引
    d = (i-1) * L + d # 全段
    d = d + 1
    print('最大值序号为：', d)  # index()函数获取索引位置
    J = d % L  # 除法取余数，这里考虑的是反向获取i，j，当前段数内序号
    if J == 0:
        J = 24
    print('最大值在当前段内序号J为：', J)
    # print('最大值在当前段段数为：', i)
    I = d // L  # 除法取整，获得i，段数
    # print('最大值在当前段的段数I为：', I)
    if J == 0:
        I = I
        print('当前段数I为：', I)
    else:
        I = I + 1
        print('当前段数I为：', I)
    j0 = J
    if j0 > 1:
        for c in range(1, j0):  # 1 to j0-1
            # print("左边和右边数据分别为：", cost_.iloc[L * I + c - 1, 2], cost_.iloc[L * I + c, 2])
            if cost_.iloc[L * (I-1) + c - 1, 2] < cost_.iloc[L * (I-1) + c, 2]:  # 右边数据大
                Dxl = Dxl + 1 # 左边计数函数
                print('Dxl=', Dxl)
    if j0 < L:
        for c in range(j0, L):  # j0 to L-1
            # print("左边和右边数据分别为：", cost_.iloc[L * I + c - 1, 2], cost_.iloc[L * I + c, 2])
            if cost_.iloc[L * (I-1) + c - 1, 2] > cost_.iloc[L * (I-1) + c, 2]:  # 左边数据大
                Dxr = Dxr + 1 # 右边计数函数
                print('Dxr=', Dxr)
    D = float(Dxl + Dxr)
    print('D = ', D)
    if (D/(2 * h)) > gama: # h为bandwidth 带宽，考虑为15*6的横坐标宽度
    # d = list(costi).index(max)
        # lmax = L * (I - 1) + ci - 1 + J  # c(i)计数函数(指示函数），确定全局Change-point位置
        X = C.iloc[d - 1, 1]
        lmax = X + ci - 1
        print('cost最大总体序号', X)
        print('                 突变点位置为：', lmax)
        area = data.iloc[lmax-10:lmax+10+1, :]
        print('                 突变点区域为：', area)
        AREA = AREA._append(area, ignore_index=True)
    AREA.to_excel('AREA.xlsx', columns=None, index=True)
'''
            if ci - 1 + J > L:
                pass    # 舍弃超过计算获得该段数的突变点
            else:
'''