"""
数据分组核心：groupby机制
分组三部曲：
    1.拆分：按指定列（如性别）将数据分成不同组
    2.应用：将每组应用统计函数（如mean()、sum()）
    3.合并：将结果整合为新的DataFrame
"""
#示例：按性别分组计算平均年龄
df.groupby('性别')['年龄'].mean()

#等价逻辑拆解:
groups = df.groupby('性别') #拆分
ages = groups['年龄']       #选择列
result = ages.mean()       #应用函数

"""
分组键的类型
分组键类型	            示例代码	                        说明
列名（字符串）	            df.groupby('部门')	            按 DataFrame 的某列分组
列名列表	                df.groupby(['部门', '性别'])	    按多列组合分组（生成层次化索引）
函数 / Series	        df.groupby(len)	                按索引长度分组（或传入自定义函数）
字典 /map	            df.groupby(mapping_dict)	    按映射关系分组（如将列名映射到组）

"""

#分组后可执行的操作
#1.聚合函数（内置统计）
# 常用聚合函数
df.groupby('部门').agg({
    '年龄': 'mean',      # 平均年龄
    '工资': 'sum',       # 总工资
    '员工ID': 'count'    # 员工数量（去重计数用nunique）
})

# 等效写法（更灵活）
df.groupby('部门')['年龄'].agg(['mean', 'min', 'max'])

#2.自定义聚合函数
# 定义分位数函数
def q25(x):
    return x.quantile(0.25)

# 应用自定义函数
df.groupby('部门')['工资'].agg(['mean', q25, lambda x: x.max()-x.min()])

#3.转换操作 transform
# 按部门填充工资缺失值为部门均值
df['工资'] = df.groupby('部门')['工资'].transform(lambda x: x.fillna(x.mean()))

# 计算每个员工工资相对于部门均值的比例
df['工资比例'] = df.groupby('部门')['工资'].transform(lambda x: x/x.mean())

#4.过滤操作 filter
# 筛选员工数量超过10人的部门
large_depts = df.groupby('部门').filter(lambda x: len(x) > 10)