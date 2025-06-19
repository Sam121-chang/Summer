
#一、Numpy矩阵运算的核心：从数组到线性代数
"""
#矩阵与数组的本质区别
Numpy数组：多维数据结构，支持任意维度（1D/2D/3D）
矩阵对象：专门用于线性代数运算的2D结构，继承自数组但功能更聚焦
"""
import numpy as np
#数组转矩阵
arr=np.array([1,2],[3,4])
mat=np.mat(arr)   #或 no.matrix(arr)

#矩阵特性
print(mat.shape) #(2,2)
print(type(mat)) # <class 'numpy.matrix'>


"""
#矩阵运算基础
"""
#1.矩阵乘法（3种方法）

#案例矩阵：
a = np.mat([1,2],[3,4])
b = np.mat([5,6],[7,8])

#method 1: @符号
c1 = a @ b

#method 2: np.matmul()函数
c2 = np.matmul(a,b)

#method 3: matrix对象的dot方法
c3 = a.dot(b)

print(c1)
print(c2)
print(c3)


#2.矩阵转置与逆

#转置
a_T = a.T
print(a_T) #[[1,3],[2,4]]

#逆矩阵（仅适用于可逆矩阵）
a_inv = np.linalg.inv(a)
print(a_inv) #[[-2.   1. ],[1.5 -0.5]]

#验证：A*A^(-1) =I
print(a @ a_inv)  #近似单位矩阵


#3.行列式与特征值分解
#行列式
det = np.linalg.det(a)  #计算｜A｜=1*4 - 2*3= -2

#特征值与特征向量
eigenvalues, eigenvectors = np.linalg.eig(a)
print(eigenvalues) #[5.0.]
print(eigenvectors) #特征向量矩阵

