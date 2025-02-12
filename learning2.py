#!/usr/bin/env python3
# 输出cost_.csv 计算ASCC统计量即cost(i,j)
# 导入pandas包并重命名为pd
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, DateFormatter
import xlrd
import time
import pandas as pd
import numpy as np
import math
import xlwt

pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.read_excel('Z.xlsx')
nrows = df.shape[0]   # 行数
#print(nrows)

M = []
N = []
for i in df['Distance']:
    M.append(i)
xmax = max(M)
xmin = min(M)
print("Distance最大值、最小值分别为：", xmax, xmin)
for i in df['Variability']:
    N.append(i)
y_max_ = max(N)
y_min_ = min(N)

print("Variability最大值、最小值分别为：", y_max_, y_min_)

xi = []
yi = []
costij = []
ci = 0
for i in range(1, nrows+1):     # for i in range(1, lc + 1):
    xi = df.iloc[i-1, 3]
    yi = df.iloc[i-1, 5]
    print(xi)
    print(yi)
    x_max = xi + 10
    x_min = xi - 10
    y_max = yi + 0.05
    y_min = yi - 0.05
    # m = 11 m取11时，代表将区域划分为10*10个子区域，所以在这里数字加1，划分15*6
    m = 7
    n = 16
    vlines = np.linspace(x_min, x_max, m)   # 散点图时间范围，划分的数量
    hlines = np.linspace(y_min, y_max, n)
    plt.vlines(vlines, min(hlines), max(hlines), colors='.25', linewidth=.75)
    plt.hlines(hlines, min(vlines), max(vlines), colors='.25', linewidth=.75)
    xs, ys = np.meshgrid(vlines[1:], hlines[:-1])     # 划分网格
    # 从左到右，从下至上依次编号并显示
    for i, (x, y) in enumerate(zip(xs.flatten(), ys.flatten())):
        #  flatten()函数返回一个一维数组
        #  zip()函数打包
        #  enumerate()函数遍历获得索引和元素值
        plt.text(x, y, str(i + 1), horizontalalignment='right', verticalalignment='bottom',)
        plt.xticks(rotation=90)
        plt.rc('font', family='Times New Roman')  # 定义坐标轴字体
    # plt.savefig('Figure_1.jpg')  # 保存图片
    # plt.show() # 展示图片
    # 统计每个子区域中点的数量
    x = []
    y = []
    n = 0
    data = pd.read_excel('Z.xlsx')  # 重新读取数据文件，数据中有x,y两列
    for row1 in data['Distance']:
        n += 1
        x.append(row1)
    for row2 in data['Variability']:
        y.append(row2)
    c = {"Distance": x, 'Variability': y}
    data = pd.DataFrame(c)
    # 使用矩阵分隔网格
    # 生成网格ID column_num等于列数，row_num等于行数
    def generalID(x, y, column_num, row_num):
        # 若在范围外的点，返回-1
        if x < x_min or x > x_max or y < y_min or y > y_max:
            return -1
        # 把范围根据列数等分切割
        column = (x_max - x_min) / column_num
        # 把范围根据行数等分切割
        row = (y_max - y_min) / row_num
        # 得到二维矩阵坐标索引，并转换为一维ID，即： 列坐标区域（向下取整）+ 1 + 行坐标区域 * 列数
        return int((x - x_min) / column) + 1 + int((y - y_min) / row) * column_num
    # 对整个区域使用 15 X 6 划分
    data['label'] = data.apply(lambda x: generalID(x['Distance'], x['Variability'], 15, 6), axis=1)
    a = data.label
    # 对给定的数组 a 的元素进行排序,默认axis = 1 按行排序， axis = 0 按列排序
    b = np.sort(a)
    D = {}    # 定义空字典D = {}
    for k in b:
        D[k] = D.get(k, 0) + 1
    for i in range(0, 91):
        if i in D:
            D[i] = D[i]
        else:
            D[i] = 0
    # print(D)
    kr = 45
    '''
    #  还没想好其规律性，关于其规律性，后续想出后重新编写这部分繁琐代码
    for j in range(0, L+1):
            for a in range(0, kr):
                lkr[a] = D[]
                rkr[a] = D[]
    '''
    lkr1 = D[85]
    rkr1 = D[90]
    lkr2 = D[86]
    rkr2 = D[89]
    lkr3 = D[87]
    rkr3 = D[88]
    lkr4 = D[79]
    rkr4 = D[84]
    lkr5 = D[80]
    rkr5 = D[83]
    lkr6 = D[81]
    rkr6 = D[82]
    lkr7 = D[73]
    rkr7 = D[78]
    lkr8 = D[74]
    rkr8 = D[77]
    lkr9 = D[75]
    rkr9 = D[76]
    lkr10 = D[67]
    rkr10 = D[72]
    lkr11 = D[68]
    rkr11 = D[71]
    lkr12 = D[69]
    rkr12 = D[70]
    lkr13 = D[61]
    rkr13 = D[66]
    lkr14 = D[62]
    rkr14 = D[65]
    lkr15 = D[63]
    rkr15 = D[64]
    lkr16 = D[55]
    rkr16 = D[60]
    lkr17 = D[56]
    rkr17 = D[59]
    lkr18 = D[57]
    rkr18 = D[58]
    lkr19 = D[49]
    rkr19 = D[54]
    lkr20 = D[50]
    rkr20 = D[53]
    lkr21 = D[51]
    rkr21 = D[52]
    lkr22 = D[43]
    rkr22 = D[48]
    lkr23 = D[44]
    rkr23 = D[47]
    lkr24 = D[45]
    rkr24 = D[46]
    lkr25 = D[37]
    rkr25 = D[42]
    lkr26 = D[38]
    rkr26 = D[41]
    lkr27 = D[39]
    rkr27 = D[40]
    lkr28 = D[31]
    rkr28 = D[36]
    lkr29 = D[32]
    rkr29 = D[35]
    lkr30 = D[33]
    rkr30 = D[34]
    lkr31 = D[25]
    rkr31 = D[30]
    lkr32 = D[26]
    rkr32 = D[29]
    lkr33 = D[27]
    rkr33 = D[28]
    lkr34 = D[19]
    rkr34 = D[24]
    lkr35 = D[20]
    rkr35 = D[23]
    lkr36 = D[21]
    rkr36 = D[22]
    lkr37 = D[13]
    rkr37 = D[18]
    lkr38 = D[14]
    rkr38 = D[17]
    lkr39 = D[15]
    rkr39 = D[16]
    lkr40 = D[7]
    rkr40 = D[12]
    lkr41 = D[8]
    rkr41 = D[11]
    lkr42 = D[9]
    rkr42 = D[10]
    lkr43 = D[1]
    rkr43 = D[6]
    lkr44 = D[2]
    rkr44 = D[5]
    lkr45 = D[3]
    rkr45 = D[4]
    if lkr45 == 0:
        if rkr45 == 0:
            lkr45 = rkr45 = 1
    if lkr44 == 0:
        if rkr44 == 0:
            lkr44 = rkr44 = 1
    if lkr43 == 0:
        if rkr43 == 0:
            lkr43 = rkr43 = 1
    if lkr42 == 0:
        if rkr42 == 0:
            lkr42 = rkr42 = 1
    if lkr41 == 0:
        if rkr41 == 0:
            lkr41 = rkr41 = 1
    if lkr40 == 0:
        if rkr40 == 0:
            lkr40 = rkr40 = 1
    if lkr39 == 0:
        if rkr39 == 0:
            lkr39 = rkr39 = 1
    if lkr38 == 0:
        if rkr38 == 0:
            lkr38 = rkr38 = 1
    if lkr37 == 0:
        if rkr37 == 0:
            lkr37 = rkr37 = 1
    if lkr36 == 0:
        if rkr36 == 0:
            lkr36 = rkr36 = 1
    if lkr35 == 0:
        if rkr35 == 0:
            lkr35 = rkr35 = 1
    if lkr34 == 0:
        if rkr34 == 0:
            lkr34 = rkr34 = 1
    if lkr33 == 0:
        if rkr33 == 0:
            lkr33 = rkr33 = 1
    if lkr32 == 0:
        if rkr32 == 0:
            lkr32 = rkr32 = 1
    if lkr31 == 0:
        if rkr31 == 0:
            lkr31 = rkr31 = 1
    if lkr30 == 0:
        if rkr30 == 0:
            lkr30 = rkr30 = 1
    if lkr29 == 0:
        if rkr29 == 0:
            lkr29 = rkr29 = 1
    if lkr28 == 0:
        if rkr28 == 0:
            lkr28 = rkr28 = 1
    if lkr27 == 0:
        if rkr27 == 0:
            lkr27 = rkr27 = 1
    if lkr26 == 0:
        if rkr26 == 0:
            lkr26 = rkr26 = 1
    if lkr25 == 0:
        if rkr25 == 0:
            lkr25 = rkr25 = 1
    if lkr24 == 0:
        if rkr24 == 0:
            lkr24 = rkr24 = 1
    if lkr23 == 0:
        if rkr23 == 0:
            lkr23 = rkr23 = 1
    if lkr22 == 0:
        if rkr22 == 0:
            lkr22 = rkr22 = 1
    if lkr21 == 0:
        if rkr21 == 0:
            lkr21 = rkr21 = 1
    if lkr20 == 0:
        if rkr20 == 0:
            lkr20 = rkr20 = 1
    if lkr19 == 0:
        if rkr19 == 0:
            lkr19 = rkr19 = 1
    if lkr18 == 0:
        if rkr18 == 0:
            lkr18 = rkr18 = 1
    if lkr17 == 0:
        if rkr17 == 0:
            lkr17 = rkr17 = 1
    if lkr16 == 0:
        if rkr16 == 0:
            lkr16 = rkr16 = 1
    if lkr15 == 0:
        if rkr15 == 0:
            lkr15 = rkr15 = 1
    if lkr14 == 0:
        if rkr14 == 0:
            lkr14 = rkr14 = 1
    if lkr13 == 0:
        if rkr13 == 0:
            lkr13 = rkr13 = 1
    if lkr12 == 0:
        if rkr12 == 0:
            lkr12 = rkr12 = 1
    if lkr11 == 0:
        if rkr11 == 0:
            lkr11 = rkr11 = 1
    if lkr10 == 0:
        if rkr10 == 0:
            lkr10 = rkr10 = 1
    if lkr9 == 0:
        if rkr9 == 0:
            lkr9 = rkr9 = 1
    if lkr8 == 0:
        if rkr8 == 0:
            lkr8 = rkr8 = 1
    if lkr7 == 0:
        if rkr7 == 0:
            lkr7 = rkr7 = 1
    if lkr6 == 0:
        if rkr6 == 0:
            lkr6 = rkr6 = 1
    if lkr5 == 0:
        if rkr5 == 0:
            lkr5 = rkr5 = 1
    if lkr4 == 0:
        if rkr4 == 0:
            lkr4 = rkr4 = 1
    if lkr3 == 0:
        if rkr3 == 0:
            lkr3 = rkr3 = 1
    if lkr2 == 0:
        if rkr2 == 0:
            lkr2 = rkr2 = 1
    if lkr1 == 0:
        if rkr1 == 0:
            lkr1 = rkr1 = 1
    cost = 0.5 * ((pow((lkr1 - rkr1), 2) / (lkr1 + rkr1))
                  + (pow((lkr2 - rkr2), 2) / (lkr2 + rkr2))
                  + (pow((lkr3 - rkr3), 2) / (lkr3 + rkr3))
                  + (pow((lkr4 - rkr4), 2) / (lkr4 + rkr4))
                  + (pow((lkr5 - rkr5), 2) / (lkr5 + rkr5))
                  + (pow((lkr6 - rkr6), 2) / (lkr6 + rkr6))
                  + (pow((lkr7 - rkr7), 2) / (lkr7 + rkr7))
                  + (pow((lkr8 - rkr8), 2) / (lkr8 + rkr8))
                  + (pow((lkr9 - rkr9), 2) / (lkr9 + rkr9))
                  + (pow((lkr10 - rkr10), 2) / (lkr10 + rkr10))
                  + (pow((lkr11 - rkr11), 2) / (lkr11 + rkr11))
                  + (pow((lkr12 - rkr12), 2) / (lkr12 + rkr12))
                  + (pow((lkr13 - rkr13), 2) / (lkr13 + rkr13))
                  + (pow((lkr14 - rkr14), 2) / (lkr14 + rkr14))
                  + (pow((lkr15 - rkr15), 2) / (lkr15 + rkr15))
                  + (pow((lkr16 - rkr16), 2) / (lkr16 + rkr16))
                  + (pow((lkr17 - rkr17), 2) / (lkr17 + rkr17))
                  + (pow((lkr18 - rkr18), 2) / (lkr18 + rkr18))
                  + (pow((lkr19 - rkr19), 2) / (lkr19 + rkr19))
                  + (pow((lkr20 - rkr20), 2) / (lkr20 + rkr20))
                  + (pow((lkr21 - rkr21), 2) / (lkr21 + rkr21))
                  + (pow((lkr22 - rkr22), 2) / (lkr22 + rkr22))
                  + (pow((lkr23 - rkr23), 2) / (lkr23 + rkr23))
                  + (pow((lkr24 - rkr24), 2) / (lkr24 + rkr24))
                  + (pow((lkr25 - rkr25), 2) / (lkr25 + rkr25))
                  + (pow((lkr26 - rkr26), 2) / (lkr26 + rkr26))
                  + (pow((lkr27 - rkr27), 2) / (lkr27 + rkr27))
                  + (pow((lkr28 - rkr28), 2) / (lkr28 + rkr28))
                  + (pow((lkr29 - rkr29), 2) / (lkr29 + rkr29))
                  + (pow((lkr30 - rkr30), 2) / (lkr30 + rkr30))
                  + (pow((lkr31 - rkr31), 2) / (lkr31 + rkr31))
                  + (pow((lkr32 - rkr32), 2) / (lkr32 + rkr32))
                  + (pow((lkr33 - rkr33), 2) / (lkr33 + rkr33))
                  + (pow((lkr34 - rkr34), 2) / (lkr34 + rkr34))
                  + (pow((lkr35 - rkr35), 2) / (lkr35 + rkr35))
                  + (pow((lkr36 - rkr36), 2) / (lkr36 + rkr36))
                  + (pow((lkr37 - rkr37), 2) / (lkr37 + rkr37))
                  + (pow((lkr38 - rkr38), 2) / (lkr38 + rkr38))
                  + (pow((lkr39 - rkr39), 2) / (lkr39 + rkr39))
                  + (pow((lkr40 - rkr40), 2) / (lkr40 + rkr40))
                  + (pow((lkr41 - rkr41), 2) / (lkr41 + rkr41))
                  + (pow((lkr42 - rkr42), 2) / (lkr42 + rkr42))
                  + (pow((lkr43 - rkr43), 2) / (lkr43 + rkr43))
                  + (pow((lkr44 - rkr44), 2) / (lkr44 + rkr44))
                  + (pow((lkr45 - rkr45), 2) / (lkr45 + rkr45)))
    print(cost)
    costij = pd.Series(cost)
    costij = costij.round(2)   #四舍五入保留两位小数
    # print(costij)
    ci += 1    # 计数函数
    print(ci)  # c(i)为计数的，最终lmax确定Change-point位置
    costij.to_csv('cost_Z.csv', mode='a')      #  保存数据