import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

df=pd.read_csv('../data/processed_titanic.csv')
X=df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked_Q','Embarked_S','HasCabin']]
y=df['Survived']

#划分训练集和测试集
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#选择模型
model=LogisticRegression()

#拟合模型
model.fit(X_train,y_train)

#训练
y_pred=model.predict(X_test)

#评估
acc = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred)
report = classification_report(y_test,y_pred)

#输出结果
print(f"模型在测试集上的准确率：{acc:4f}")
print("混淆矩阵： ")
print(cm)
print("分类报告： ")
print(report)


#手动获取混淆矩阵核心指标（TP、FP、FN、TN)
from sklearn.metrics import confusion_matrix

# 生成混淆矩阵
cm = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = cm.ravel()

print("True Negative:", TN)
print("False Positive:", FP)
print("False Negative:", FN)
print("True Positive:", TP)


#手动计算Precision、Recall、F1-score
# 手动计算指标
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"手动计算的 Precision: {precision:.4f}")
print(f"手动计算的 Recall: {recall:.4f}")
print(f"手动计算的 F1-score: {f1_score:.4f}")


#对比sklearn自带效果，验证你的推导正确性
from sklearn.metrics import classification_report

print("Sklearn的分类报告：")
print(classification_report(y_test, y_pred, digits=4))


#可视化混淆矩阵
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

"""

· Titanic 项目模型评估报告：混淆矩阵 + Precision / Recall 分析笔记**



---


> **目的：** 在完成模型训练基础上，深入理解分类器性能的评估方式，学会将“准确率”解剖成更细致的评价指标：**Precision / Recall / F1-score**，并理解其适用场景、优化策略与未来模型对比的基准。

---

## 🔍 一、模型基础表现（准确率）

```text
模型在测试集上的准确率：0.821229（≈82.12%）
```

**解读：** 该值衡量了模型整体的预测正确率，但在不平衡数据、不同类型错误代价不等时并不可靠，必须进一步拆解。

---

## 🧱 二、混淆矩阵结果分析

```text
[[TN=91  FP=14]
 [FN=18  TP=56]]
```

| 实际\预测 | 预测0（没生存）   | 预测1（生存）    |
| ----- | ---------- | ---------- |
| 实际0   | TN=91（预测对） | FP=14（误杀）  |
| 实际1   | FN=18（漏救）  | TP=56（预测对） |

**解读关键词：**

* **True Positive (TP)**: 实际生还，模型预测也为生还 → ✅ 成功识别
* **False Positive (FP)**: 实际未生还，但模型预测为生还 → ❌ 虚假希望
* **False Negative (FN)**: 实际生还，模型预测为未生还 → ❌ 错失救援
* **True Negative (TN)**: 实际未生还，模型预测也为未生还 → ✅ 正确识别

---

## 📐 三、核心评估指标计算（手动推导 + sklearn对比）

| 指标             | 计算公式           | 手动值    | Sklearn值 | 解释                          |
| -------------- | -------------- | ------ | -------- | --------------------------- |
| Precision（精确率） | TP / (TP + FP) | 0.8000 | 0.8000   | 在模型说“能活”的人中，有多少是真的          |
| Recall（召回率）    | TP / (TP + FN) | 0.7568 | 0.7568   | 在真正能活的人中，有多少被模型识别出来         |
| F1-score       | 2PR/(P+R)      | 0.7778 | 0.7778   | 平衡 Precision 与 Recall 的综合指标 |

---

## 🎯 四、从“准确率”到“指标策略”的迁移认知

| 评价角度            | 指标        | 用于优化哪些模型情境                  |
| --------------- | --------- | --------------------------- |
| 不可错过任何正样本（宁可错杀） | Recall    | 医疗诊断、金融风控（高 FN 代价）          |
| 不可误报（宁可少报）      | Precision | 法律、刑事系统（高 FP 代价）            |
| 追求整体平衡          | F1-score  | 泛用型任务，兼顾 Precision / Recall |

---

## 🧠 五、能力迁移：下一阶段项目的指标意识

本次评估建立了一个完整分类器分析链条，为接下来房价预测项目（回归问题）、决策树等模型提供了：

* ✅ **思维模板**：如何从模型结构 → 评估行为 → 定位问题；
* ✅ **指标基础**：了解分类任务的多个目标权衡；
* ✅ **动手训练**：从黑盒调用到手动验证，掌握了“计算过程”。

---

 附录：代码补档（完整评估流程）

<details>
<summary>点击展开代码</summary>

```python
from sklearn.metrics import confusion_matrix, classification_report

# 混淆矩阵生成
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

# 手动计算指标
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)

# 输出结果
print(f"True Negative: {TN}")
print(f"False Positive: {FP}")
print(f"False Negative: {FN}")
print(f"True Positive: {TP}")

print(f"手动计算的 Precision: {precision:.4f}")
print(f"手动计算的 Recall: {recall:.4f}")
print(f"手动计算的 F1-score: {f1_score:.4f}")

print("Sklearn的分类报告：")
print(classification_report(y_test, y_pred, digits=4))

"""
#生成结果
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
X_test_final = test_df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked_Q','Embarked_S','HasCabin']]
test_df["Survived"] = model.predict(X_test_final)

# 4. 生成提交文件（只需 PassengerId 和 Survived）
submission = test_df[["PassengerId", "Survived"]]
submission.to_csv("submission.csv", index=False)

print("✅ submission.csv 预测文件已成功生成！")
