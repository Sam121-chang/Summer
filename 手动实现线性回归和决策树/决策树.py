import numpy as np

X = np.array([2,3,10,19,25])
y = np.array([0,0,1,1,1]) #二分类：0 or 1

def gini(y): #计算基尼不纯度，越小表示越纯（分类清晰）
    classes = np.unique(y)
    impurity = 1
    for c in classes:
        p = np.sum(y==c)/len(y)
        impurity -= p**2
    return impurity

#暴力法找最优划分点
best_gini = 1
best_spilt= None
for spilt in X:
    left_mask = X<spilt
    right_mask = X>=spilt
    gini_spilt=(np.sum(left_mask) * gini(y[left_mask]) + np.sum(right_mask) * gini(y[right_mask])) / len(y)
    if gini_spilt < best_gini:
        best_gini = gini_spilt
        best_spilt = spilt
        #枚举所有可能的分割点spilt：• 把样本分成 < split 和 >= split 两类，对每一类计算基尼不纯度，再加权平均 找到最小的不纯度作为“最佳切分点”
print("最优划分点是：",best_spilt)