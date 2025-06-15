import pandas as pd
#基础用法：读取目录下的csv文件
df = pd.read_csv('employee.csv')
#进阶：指定分隔符（如 | 分隔的文件）、编码（处理中文乱码）、跳过首行
df = pd.read_csv('employee.csv',sep='|',encoding=
                 'utf-8')
#读 Excle：需特定工作表，sheet_name 可为索引/名称
df_excel = pd.read_excel('employee.xslx',sheet_name='Sheet1')

"""
sep：文件分隔符（CSV常用，也可以自定义如\t表符）
encoding：解决中文乱码（utf-9/gbk常用）
skiprows：跳过前n行（比如文件首行是广告）
sheet_name：Excel中选工作表（填索引0/名称‘销售数据’）
"""
 #读取后必做检查  举例：若 df.dtypes 里某列本该是数字，却显示 object（字符串 ），可能是数据里混了文字，后续需处理
print(df.head())  # 看前 5 行，快速预览数据
print(df.tail(3))  # 看后 3 行，检查末尾数据
print(df.shape)  # 输出 (行数, 列数)，判断数据规模
print(df.dtypes)  # 看各列数据类型，比如是否有字符串存成数字
print(df.isnull().sum())  # 统计每列缺失值，决定清洗策略


#数据筛选
#（一）按列筛选：提取指定列
# 1.单/多列提取（方括号语法）
#单列（返回Series类型）
name_series = df['姓名']
#多列（返回DataFrame类型，注意双层方括号）
subset = df[['姓名','年龄','工资']]
"""
逻辑：外层方括号是‘选列操作‘，内层[列表]指明想要哪些列，
类似从表格里挑几列复制
"""
#2.用loc按列名筛选（更灵活）
#选所有行，列名含‘收入’的列（支持模糊匹配思路：遍历列名判断）
income_cols = df.loc[:,df.xolums.str.contains('收入')]
#固定列名筛选（推荐，简单场景直接用）
fixed_subset=df.loc[:,['姓名','部门']]
"""
优势：loc 支持按列名逻辑筛选（如包含某字、以某字结尾 ），比纯方括号更灵活
"""
"""按行筛选：条件查询
1. 基础条件（布尔索引 ）"""

# 筛选年龄 > 30 岁的行
adults = df[df['年龄'] > 30]
# 多条件（注意用 & | 连接，条件加括号）
cond = (df['年龄'] > 30) & (df['工资'] < 10000)
middle_age_low_salary = df[cond]
"""原理：df['年龄'] > 30 生成布尔 Series（每行对应 True/False ），用它当 “过滤器”，只保留 True 的行。"""

#2. 用 loc 结合条件（推荐 ）
# 语法：loc[行条件, 列条件]
high_income = df.loc[df['工资'] > 20000, ['姓名', '工资']]
# 复杂条件：年龄 25 - 40 岁且部门是销售
cond = (df['年龄'].between(25, 40)) & (df['部门'] == '销售')
sales_people = df.loc[cond, :]  # : 表示选所有列
"""好处：loc 能同时控制 “哪些行” 和 “哪些列”，一步到位，代码更清晰。"""

#3. 按索引筛选（iloc ）
# 选第 2 - 4 行（索引从 0 开始，左闭右开），第 1 - 3 列
subset_by_pos = df.iloc[1:4, 0:3]
# 选第 0 行、第 2 行，第 0 列、第 3 列
specific_cells = df.iloc[[0, 2], [0, 3]]
"""适用场景：知道数据的位置索引（比如前 5 行、第 3 列 ），而非列名时用，像处理无表头的纯数据文件。"""

#（三）进阶筛选技巧
#1. 用 query 简化条件
# 简单条件
result = df.query('工资 > 15000 and 部门 == "研发"')
# 动态传变量（用 @ 引用 Python 变量）
min_salary = 10000
dynamic_result = df.query('工资 > @min_salary')
"""优势：条件写起来像自然语言，少打很多 df[]，复杂条件更易读。"""
#2. 筛选含特定内容的行

# 姓名列包含 '张' 字
zhang_family = df[df['姓名'].str.contains('张')]
# 工资列是整数（过滤掉带小数的异常值）
integer_salary = df[df['工资'].apply(lambda x: isinstance(x, int))]

"""应用：处理文本列（如姓名、地址 ）时，筛选含关键词的行；或过滤数据类型异常的行。"""

