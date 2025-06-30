#%% md
# 数据基本情况：
#%%
import pandas as pd
import numpy as np

df=pd.read_csv('train.csv')
df.info()
df.head()
df.describe()
df.isnull().sum()
#%% md
# 以下是单变量分析部分：
#%%
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Pclass', data=df)
plt.title('Pclass Distribution Map')
plt.show()
#%%
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Sex', data=df)
plt.title('Sex Distribution Map')
plt.show()
#%%
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df['Age'],kde=True)
plt.title('Age Distribution Map')
plt.show()
#%%
import seaborn as sns
import matplotlib.pyplot as plt

sns.swarmplot(x='SibSp', data=df,size=0.3)
plt.title('SibSp Distribution Map')
plt.show()
#%%
import seaborn as sns
import matplotlib.pyplot as plt

sns.swarmplot(x='Parch',data=df,size=0.3)
plt.title('Parch Distribution Map')
plt.show()
#%%
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df['Fare'],kde=True)
plt.title('Fare Distribution Map')
plt.show()

#%%
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Embarked',data=df)
plt.title('Embarked Distribution Map')
plt.show()
#%% md
# 以下是双变量分析部分：
# 
#%%
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Pclass',hue='Survived',data=df)
plt.title('Pclass vs Survived Distribution Map')
plt.show()
#%%
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Sex',hue='Survived',data=df)
plt.title('Sex vs Survived Distribution Map')
plt.show()
#%%
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(data=df,x='Age',hue='Survived',kde=True)
plt.title('Age vs Survived Distribution Map')
plt.show()
#%%
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(data=df,x='Fare',hue='Survived',kde=True)
plt.title('Fare vs Survived Distribution Map')
plt.show()
#%%
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=df,x='Embarked', hue='Survived')
plt.title('Embarked vs Survived Distribution Map')
plt.show()
#%%
import seaborn as sns
import matplotlib.pyplot as plt

sns.barplot(data=df,x='SibSp', y='Survived')
plt.title('SibSp vs Survived Distribution Map')
plt.show()

sns.barplot(data=df,x='Parch', y='Survived')
plt.title('Parch vs Survived Distribution Map')
plt.show()
#%% md
# 以下是多变量组合
# 
#%%
import seaborn as sns
import matplotlib.pyplot as plt

sns.catplot(x="Pclass", y="Survived", hue="Sex", col="Embarked", data=df, kind="point")
plt.title('Multivariate combination vs Survived Distribution Map')
plt.show()
#%%
#年龄vs性别
import seaborn as sns
import matplotlib.pyplot as plt

sns.violinplot(data=df,x="Sex", y="Age", hue="Survived", split=True)
plt.title('Age vs Sex Distribution Map')
plt.show()
#%%
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(df.isnull(), cbar=False)