import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("processed_train.csv")

X = df.drop(columns=['SalePrice'])
y = df['SalePrice']

y_bins = pd.qcut(y,q=3,labels=['Low','Mid','High'])


from sklearn.decomposition import PCA

pca=PCA(n_components=2)
X_pca=pca.fit_transform(X)

explained = pca.explained_variance_ratio_
print(f"PCA前两维解释了总方差的比例:{explained.sum():.4f}")


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))
sns.scatterplot(x=X_pca[:,0],y=X_pca[:,1],hue=y_bins,palette='coolwarm')
plt.xlabel("PAC1")
plt.ylabel("PAC2")
plt.show()

from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# 降至3维
pca3 = PCA(n_components=3)
X_pca3 = pca3.fit_transform(X)

# 3D可视化
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca3[:,0], X_pca3[:,1], X_pca3[:,2], c=y, cmap="coolwarm", alpha=0.7)
fig.colorbar(scatter, ax=ax, label="SalePrice")
ax.set_title("PCA 3D Projection")
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.set_zlabel("PCA3")
plt.tight_layout()
plt.savefig("pca_3d_visualization.png")
plt.show()
