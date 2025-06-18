import numpy as np

# =====================================
# 一、数组创建练习
# =====================================

# 1. 基础创建方法
print("=== 基础创建方法 ===")

# 创建一维数组
arr1 = np.array([1, 2, 3, 4, 5])
print("一维数组:\n", arr1)

# 创建二维数组（注意嵌套列表）
arr2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("二维数组:\n", arr2)

# 创建全0数组（参数为元组，表示形状）
zeros_arr = np.zeros((3, 4))  # 3行4列
print("全0数组:\n", zeros_arr)

# 创建全1数组
ones_arr = np.ones((2, 5))  # 2行5列
print("全1数组:\n", ones_arr)

# 创建指定值的数组
full_arr = np.full((3, 3), 7)
print("全7数组:\n", full_arr)

# 创建等差数列（类似Python的range，但支持浮点数）
arange_arr = np.arange(0, 10, 2)  # 从0到10(不包含)，步长2
print("等差数列:\n", arange_arr)

# 创建等比数列（在指定范围内生成均匀分布的数）
linspace_arr = np.linspace(0, 1, 5)  # 从0到1，均匀分成5个点
print("等比数列:\n", linspace_arr)

# 创建随机数组（[0,1)之间的均匀分布）
random_arr = np.random.rand(3, 3)  # 3x3的随机数
print("随机数组:\n", random_arr)

# 创建标准正态分布的随机数组
normal_arr = np.random.normal(0, 1, (2, 2))  # 均值0，标准差1
print("正态分布随机数组:\n", normal_arr)

# =====================================
# 二、数组属性练习
# =====================================
print("\n=== 数组属性练习 ===")

print("数组形状:", arr2.shape)  # 输出(3, 3)，表示3行3列
print("数组维度:", arr2.ndim)   # 输出2，表示二维数组
print("数组元素类型:", arr2.dtype)  # 输出int64
print("数组元素个数:", arr2.size)  # 输出9
print("每个元素占用的字节数:", arr2.itemsize)  # 输出8（64位整数）

# =====================================
# 三、数组索引与切片练习
# =====================================
print("\n=== 数组索引与切片练习 ===")

# 一维数组索引
print("一维数组索引:")
print("第一个元素:", arr1[0])  # 输出1
print("最后一个元素:", arr1[-1])  # 输出5
print("前三个元素:", arr1[:3])  # 输出[1 2 3]
print("后两个元素:", arr1[-2:])  # 输出[4 5]

# 二维数组索引（行,列）
print("\n二维数组索引:")
print("第二行第三列元素:", arr2[1, 2])  # 输出6（索引从0开始）
print("第一行所有元素:", arr2[0, :])    # 输出[1 2 3]
print("第三列所有元素:", arr2[:, 2])    # 输出[3 6 9]
print("前两行，后两列:", arr2[:2, 1:])  # 输出[[2 3] [5 6]]

# 复杂切片示例
print("\n复杂切片:")
print("行倒序:", arr2[::-1, :])  # 上下翻转
print("列倒序:", arr2[:, ::-1])  # 左右翻转
print("间隔取值:", arr2[::2, ::2])  # 每隔一行一列取值

# =====================================
# 四、布尔索引练习
# =====================================
print("\n=== 布尔索引练习 ===")

# 创建布尔条件
bool_idx = arr2 > 5  # 返回一个布尔数组，形状与arr2相同
print("布尔索引数组:\n", bool_idx)

# 使用布尔索引筛选元素（返回一维数组）
print("使用布尔索引筛选元素:", arr2[bool_idx])  # 输出所有大于5的元素

# 直接在索引中使用条件
print("直接筛选大于5的元素:", arr2[arr2 > 5])  # 更简洁的写法

# 多条件筛选（使用&和|，注意加括号）
print("筛选大于2且小于8的元素:", arr2[(arr2 > 2) & (arr2 < 8)])

# =====================================
# 五、数组操作练习
# =====================================
print("\n=== 数组操作练习 ===")

# 改变数组形状（元素总数必须相同）
reshaped_arr = arr2.reshape(1, 9)  # 转为1行9列
print("重塑后的数组:\n", reshaped_arr)

# 展平数组（转为一维）
flattened_arr = arr2.flatten()
print("展平后的数组:", flattened_arr)

# 数组转置（行变列，列变行）
transposed_arr = arr2.T
print("转置后的数组:\n", transposed_arr)

# 数组拼接
arr3 = np.array([[10, 11, 12]])
concatenated_arr = np.concatenate((arr2, arr3), axis=0)  # 按行拼接
print("按行拼接后的数组:\n", concatenated_arr)

concatenated_arr2 = np.concatenate((arr2, arr3.T), axis=1)  # 按列拼接
print("按列拼接后的数组:\n", concatenated_arr2)

# =====================================
# 六、数学运算练习
# =====================================
print("\n=== 数学运算练习 ===")

# 元素级加法
add_result = arr2 + 10
print("每个元素加10:\n", add_result)

# 元素级乘法
mult_result = arr2 * 2
print("每个元素乘2:\n", mult_result)

# 两个数组相加（对应元素相加）
arr4 = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
sum_result = arr2 + arr4
print("两个数组相加:\n", sum_result)

# 矩阵乘法（注意与元素级乘法的区别）
matmul_result = np.matmul(arr2, arr4)
# 也可以用 @ 符号：arr2 @ arr4
print("矩阵乘法结果:\n", matmul_result)

# 其他数学函数
print("平方根:\n", np.sqrt(arr2))
print("指数:\n", np.exp(arr2))
print("对数:\n", np.log(arr2))
print("三角函数:\n", np.sin(arr2))

# =====================================
# 七、统计函数练习
# =====================================
print("\n=== 统计函数练习 ===")

print("数组总和:", np.sum(arr2))  # 所有元素求和
print("每列的和:", np.sum(arr2, axis=0))  # 按列求和（结果是一维数组）
print("每行的和:", np.sum(arr2, axis=1))  # 按行求和

print("数组平均值:", np.mean(arr2))  # 所有元素求平均
print("每列的平均值:", np.mean(arr2, axis=0))
print("每行的平均值:", np.mean(arr2, axis=1))

print("数组最大值:", np.max(arr2))  # 所有元素求最大值
print("每列的最大值:", np.max(arr2, axis=0))
print("每行的最大值:", np.max(arr2, axis=1))

print("数组最小值:", np.min(arr2))  # 所有元素求最小值
print("每列的最小值:", np.min(arr2, axis=0))
print("每行的最小值:", np.min(arr2, axis=1))

print("标准差:", np.std(arr2))  # 所有元素求标准差
print("方差:", np.var(arr2))    # 所有元素求方差