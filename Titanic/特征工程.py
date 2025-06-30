"""
特征工程 =把原始世界转化为“算法能理解的语言结构”
因为机器学习背后的假设是，每个输入特征都必须是：
                            数值型
                            无缺失
                            结构对齐
                            尺度一致
"""
# Titanic 特征工程完整处理脚本
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 读取原始数据
df = pd.read_csv("train.csv")

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

# 5. 输出处理后的数据
df.to_csv("processed_titanic.csv", index=False)
print("特征工程完成，文件已保存为 processed_titanic.csv")




'''
| 项目      | 操作说明                             |
| ------- | -------------------------------- |
| 清洗缺失值   | Age、Embarked 填补；Cabin → HasCabin |
| 类别变量编码  | Sex = 0/1；Embarked → OneHot      |
| 数值标准化   | Age, Fare 用 StandardScaler       |
| 产出建模数据集 | df\_cleaned，准备用于机器学习模型训练         |
H

现在的df拥有这样的结构：
    无缺失值
    所有字段都是数字
    字段具有解释性(如HasCabin)
    Age、Fare数值在同一量纲内
    Sex/Embarked转换为机器理解的格式

'''



