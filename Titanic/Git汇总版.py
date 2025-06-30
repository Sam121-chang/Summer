# 数据基本情况与探索性分析部分
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# 读取原始数据
df = pd.read_csv('train.csv')

# 数据基本情况查看
print("数据基本情况：")
df.info()
print(df.head())
print(df.describe())
print(df.isnull().sum())

# 单变量分析部分
# Pclass分布
sns.countplot(x='Pclass', data=df)
plt.title('Pclass Distribution Map')
plt.show()

# Sex分布
sns.countplot(x='Sex', data=df)
plt.title('Sex Distribution Map')
plt.show()

# Age分布
sns.histplot(df['Age'], kde=True)
plt.title('Age Distribution Map')
plt.show()

# SibSp分布
sns.swarmplot(x='SibSp', data=df, size=0.3)
plt.title('SibSp Distribution Map')
plt.show()

# Parch分布
sns.swarmplot(x='Parch', data=df, size=0.3)
plt.title('Parch Distribution Map')
plt.show()

# Fare分布
sns.histplot(df['Fare'], kde=True)
plt.title('Fare Distribution Map')
plt.show()

# Embarked分布
sns.countplot(x='Embarked', data=df)
plt.title('Embarked Distribution Map')
plt.show()

# 双变量分析部分
# Pclass与Survived关系
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Pclass vs Survived Distribution Map')
plt.show()

# Sex与Survived关系
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Sex vs Survived Distribution Map')
plt.show()

# Age与Survived关系
sns.histplot(data=df, x='Age', hue='Survived', kde=True)
plt.title('Age vs Survived Distribution Map')
plt.show()

# Fare与Survived关系
sns.histplot(data=df, x='Fare', hue='Survived', kde=True)
plt.title('Fare vs Survived Distribution Map')
plt.show()

# Embarked与Survived关系
sns.countplot(data=df, x='Embarked', hue='Survived')
plt.title('Embarked vs Survived Distribution Map')
plt.show()

# SibSp与Survived关系
sns.barplot(data=df, x='SibSp', y='Survived')
plt.title('SibSp vs Survived Distribution Map')
plt.show()

# Parch与Survived关系
sns.barplot(data=df, x='Parch', y='Survived')
plt.title('Parch vs Survived Distribution Map')
plt.show()

# 多变量组合分析
sns.catplot(x="Pclass", y="Survived", hue="Sex", col="Embarked", data=df, kind="point")
plt.title('Multivariate combination vs Survived Distribution Map')
plt.show()

# 年龄与性别
sns.violinplot(data=df, x="Sex", y="Age", hue="Survived", split=True)
plt.title('Age vs Sex Distribution Map')
plt.show()

# 查看缺失值热力图
sns.heatmap(df.isnull(), cbar=False)
plt.show()

# 特征工程部分
# 1. 缺失值处理
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Cabin 缺失太多，不能直接填充 => 新建结构性变量 HasCabin（有无舱位）
df["HasCabin"] = df["Cabin"].notnull().astype(int)

# 2. 删除无关或不适合作为特征的字段
df.drop(columns=["Name", "Ticket", "Cabin", "PassengerId"], inplace=True)

# 3. 编码处理
# Sex 映射为 0/1
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Embarked 使用 One-Hot 编码，去掉第一列防止共线性
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# 4. 数值标准化（Age 和 Fare 的量纲差异较大）
scaler = StandardScaler()
df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])

# 模型训练与评估部分
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S', 'HasCabin']]
y = df['Survived']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 选择模型
model = LogisticRegression()

# 拟合模型
model.fit(X_train, y_train)

# 训练
y_pred = model.predict(X_test)

# 评估
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# 输出结果
print(f"模型在测试集上的准确率：{acc:.4f}")
print("混淆矩阵： ")
print(cm)
print("分类报告： ")
print(report)

# 手动获取混淆矩阵核心指标（TP、FP、FN、TN)
TN, FP, FN, TP = cm.ravel()

print("True Negative:", TN)
print("False Positive:", FP)
print("False Negative:", FN)
print("True Positive:", TP)

# 手动计算Precision、Recall、F1-score
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"手动计算的 Precision: {precision:.4f}")
print(f"手动计算的 Recall: {recall:.4f}")
print(f"手动计算的 F1-score: {f1_score:.4f}")

# 对比sklearn自带效果，验证推导正确性
print("Sklearn的分类报告：")
print(classification_report(y_test, y_pred, digits=4))

# 可视化混淆矩阵
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 生成提交文件部分
# 1. 读取测试集数据
test_df = pd.read_csv('test.csv')

# 2. 和训练集一样处理：缺失值 + 特征工程
test_df["Age"].fillna(df["Age"].median(), inplace=True)
test_df["Fare"].fillna(df["Fare"].median(), inplace=True)
test_df["HasCabin"] = test_df["Cabin"].notnull().astype(int)

test_df["Sex"] = test_df["Sex"].map({'male': 0, 'female': 1})
test_df = pd.get_dummies(test_df, columns=['Embarked'], drop_first=True)

# 有些测试集中可能没有某些 Embarked 类别，补齐缺失列
for col in ['Embarked_Q', 'Embarked_S']:
    if col not in test_df.columns:
        test_df[col] = 0

# 标准化 Age 和 Fare
test_df[["Age", "Fare"]] = scaler.transform(test_df[["Age", "Fare"]])

# 3. 提取特征列并进行预测
X_test_final = test_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S', 'HasCabin']]
test_df["Survived"] = model.predict(X_test_final)

# 4. 生成提交文件（只需 PassengerId 和 Survived）
submission = test_df[["PassengerId", "Survived"]]
submission.to_csv("submission.csv", index=False)

print("✅ submission.csv 预测文件已成功生成！")