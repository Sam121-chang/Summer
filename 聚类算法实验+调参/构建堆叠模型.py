import pandas as pd
from sklearn.model_selection import train_test_split

# 读取处理后的数据
df = pd.read_csv("processed_train.csv")

# 拆分特征和标签
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# 训练集/验证集划分
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV

# 参数网格
alphas = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]

# Ridge 回归
ridge = Ridge()
ridge_grid = GridSearchCV(ridge, {"alpha": alphas}, cv=5, scoring="neg_root_mean_squared_error")
ridge_grid.fit(X_train, y_train)
print("Ridge最优参数:", ridge_grid.best_params_)
print("Ridge最优RMSE（负数）:", ridge_grid.best_score_)

# Lasso 回归
lasso = Lasso(max_iter=10000)
lasso_grid = GridSearchCV(lasso, {"alpha": alphas}, cv=5, scoring="neg_root_mean_squared_error")
lasso_grid.fit(X_train, y_train)
print("Lasso最优参数:", lasso_grid.best_params_)
print("Lasso最优RMSE（负数）:", lasso_grid.best_score_)


from sklearn.ensemble import StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 基学习器
base_models = [
    ("ridge", Ridge(alpha=ridge_grid.best_params_["alpha"])),
    ("tree", DecisionTreeRegressor(max_depth=4)),
    ("knn", KNeighborsRegressor(n_neighbors=5))
]

# 元模型
meta_model = Ridge()

# 堆叠模型
stack_model = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=5)
stack_model.fit(X_train, y_train)

# 预测 + 评估
y_pred = stack_model.predict(X_val)
rmse = mean_squared_error(y_val, y_pred, squared=False)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print("Stacking回归器性能：")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.4f}")

