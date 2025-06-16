#高级统计操作
#（一）透视表与交叉表
#1. 透视表（pivot_table）
# 按部门和性别计算平均工资
df.pivot_table(
    values='工资',       # 要聚合的值
    index='部门',        # 行索引
    columns='性别',      # 列索引
    aggfunc='mean',      # 聚合函数
    margins=True         # 添加总计行/列
)
#2. 交叉表（crosstab）
# 统计部门与职位的分布
pd.crosstab(
    index=df['部门'],    # 行
    columns=df['职位'],  # 列
    margins=True         # 添加总计
)
#（二）时间序列分组（resample）
# 假设df有日期列'date'，且已设为索引
# 按月份统计销售额
monthly_sales = df.resample('M')['销售额'].sum()

# 按季度计算平均值，向前填充缺失值
quarterly_data = df.resample('Q').mean().ffill()
#（三）窗口函数（rolling）

# 计算7天移动平均
df['7天移动平均'] = df['销售额'].rolling(window=7).mean()

# 计算每行相对于前3行的增长率
df['增长率'] = df['销售额'].pct_change(periods=3)


#性能优化与常见问题
#（一）处理大型数据集的技巧

# 1. 使用chunksize分批处理
for chunk in pd.read_csv('large_data.csv', chunksize=1000):
    chunk_stats = chunk.groupby('category')['value'].sum()
    # 累加结果或写入文件

# 2. 用Cython加速自定义聚合函数
import cython

@cython.compile
def fast_mean(x):
    return sum(x) / len(x)

df.groupby('key')['value'].agg(fast_mean)
"""
（二）常见问题排查
问题场景	可能原因	解决方案
分组后结果缺失某些组	某些组数据全为 NaN	使用dropna=False参数保留空组
聚合结果列名混乱	使用多函数聚合	用rename()重命名结果列
性能低下	数据量大且未优化	使用category类型替代字符串列
时间分组报错	日期列未转换为 datetime 类型	使用pd.to_datetime()转换日期格式
五、关键知识点速查表
操作类型	核心函数 / 方法	关键参数说明
分组	df.groupby('key')	by：分组依据（列名 / 函数 / 索引等）
聚合	agg(['mean', 'sum'])	可传入自定义函数或字典指定列聚合方式
转换	transform(func)	返回与原数据等长的转换结果
过滤	filter(lambda x: len(x)>5)	保留满足条件的组
透视表	pivot_table()	index/columns：行列索引
时间重采样	resample('M')	rule：时间频率（'D'/'W'/'M' 等）
窗口函数	rolling(window=3)	window：窗口大小
通过掌握这些分组与统计技巧，可高效完成从基础数据清洗到复杂业务分析的全流程，建议结合实际数据集（如 Kaggle 的电商数据）反复练习，形成条件反射式的数据分析思维。
"""