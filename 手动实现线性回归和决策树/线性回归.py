#假设我们有一组二维数据：x为特征，y为标签
import numpy as np

X=np.array([1,2,3,4,5])
y=np.array([2,4,6,8,10])

#这是最简单的线性关系：y=2x  我们希望模型学到这个规律

#添加偏置项b，构造X矩阵
#np.ones((len(X),1))是生成一列全1，用于偏置项b；
#np.c_是列拼接操作，最终X_b变成形如：[[1,1],[1,2],...,[1,5]]
X_b = np.c_[np.ones((len(X),1)),X]

#最小二乘法解析拆解：theta = (X^T X)^(-1) X^T y
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
#X_b.dot(x_b)是求XᵀX;• np.linalg.inv(...) 是求逆矩阵;• 最终结果 theta_best 是 [b, w] 形式，即 y = wx + b

print("模型参数 theta：",theta_best)