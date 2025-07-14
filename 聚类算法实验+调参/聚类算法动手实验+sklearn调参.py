import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

#加载 Iris 数据集
iris = load_iris()
X = iris.data
y = iris.target

#标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#初步聚类
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

#降维 PCA 方便可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

#绘图
plt.figure(figsize=(12,5))

#左图：真实标签
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:,0], X_pca[:,1],c=cluster_labels,cmap='viridis',s=50)
plt.title('聚类')
plt.show()


wcss=[]
K_range=range(1,11)

for k in K_range:
    kmeans = KMeans(n_clusters=k,random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_) #Inertia_ : SEE组内方差和

#画图
plt.plot(K_range,wcss,marker='o')
plt.title("肘部法则：选择最佳K")
plt.xlabel("簇数K")
plt.ylabel("组内误差（SEE）")
plt.grid(True)
plt.show()


silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

plt.plot(range(2, 11), silhouette_scores, marker='s', color='orange')
plt.title("轮廓系数评价")
plt.xlabel("簇数K")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()

# 改进型模型：KMeans++
kmeans = KMeans(
    n_clusters=3,
    init='k-means++',       # 初始化策略
    n_init=20,              # 初始中心选择重复次数（防止局部最优）
    max_iter=500,           # 最大迭代次数
    tol=1e-4,               # 收敛容忍度
    random_state=42
)

cluster_labels = kmeans.fit_predict(X_scaled)
print("最终聚类中心：", kmeans.cluster_centers_)
