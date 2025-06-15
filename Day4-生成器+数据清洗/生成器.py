"""
生成器：数据管道的「懒汉工厂」
1. 为什么需要生成器？
传统列表的痛点：
当数据量极大时（如 10GB 日志），列表一次性加载会导致内存溢出。

示例：读取大文件时，传统方法需将所有行存入列表：
with open('huge_file.txt') as f:
    lines = f.readlines()  # 一次性加载全部行，可能崩溃

生成器的解决方案：
按需生成数据：每次只生成需要的值，类似工厂的「订单式生产」。
节省内存：生成器对象占用固定内存（无论数据量多大）。

"""
#生成器的两种解决方案

#1.生成器表达式（小括号）
# 传统列表：一次性创建所有元素
squares_list = [x**2 for x in range(1000)]  # 占用大量内存

# 生成器表达式：按需生成元素
squares_gen = (x**2 for x in range(1000))  # 占用固定内存

print(next(squares_gen))  # 输出：0（生成第一个值）
print(next(squares_gen))  # 输出：1（生成第二个值）

#2.生成器函数（yield关键字）
def countdown(n):
    while n > 0:
        yield n  # 暂停函数，返回当前值
        n -= 1

# 使用生成器函数
gen = countdown(3)
print(next(gen))  # 输出：3
print(next(gen))  # 输出：2
print(next(gen))  # 输出：1

"""
生成器的核心机制：惰性求值
执行流程：
调用生成器函数时，函数体不立即执行，而是返回生成器对象。
每次调用next()时，函数执行到yield语句，返回值并暂停。
再次调用next()时，从暂停处继续执行，直到下一个yield或结束。
生活类比：
想象你在餐厅点菜，生成器就像厨师：
传统列表是「一次性做好所有菜」，占用大量盘子；
生成器是「你点一道，厨师做一道」，节省空间且灵活。
"""

#实战
#1.处理大文件
def read_large_file(file_path):
    with open(file_path) as f:
        for line in f:  # 文件对象本身是迭代器，逐行读取
            yield line

# 逐行处理10GB文件，内存占用恒定
for line in read_large_file('huge_data.csv'):
    process(line)

#2.无限序列生成
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# 生成斐波那契数列的前10个数
gen = fibonacci()
for _ in range(10):
    print(next(gen))  # 输出：0, 1, 1, 2, 3, 5, 8, 13, 21, 34

