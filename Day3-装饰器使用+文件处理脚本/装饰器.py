"""
核心作用：不修改原函数代码，添加新功能（如日志、缓存、权限校验）
实现代码复用，避免功能重复开发
"""
#基础语法：

def decorator(func):
    def wrapper(*args, **kwargs):
#新增功能（前）
        result = func(*args, **kwargs)
#新增功能（后）
        return result
    return wrapper

@decorator
#等价于 func = decorator
def targer_function():
    pass

#案例：日志装饰器
def log(func):
    def wrapper(*args):
        print(f"执行函数：{func.__name__},参数：{args}")
        return func(*args)
    return wrapper

@log
def add(a,b):
    return a + b

add(3,5) #输出：执行函数：add, 参数：（3，5）

"""
装饰器本质	
高阶函数 + 闭包，@语法糖是函数嵌套调用的语法简化

装饰器参数	
带参数装饰器需多一层函数嵌套（如def decorator(arg): def wrapper(func):...）
"""

#练习：编写一个计算函数执行时间的装饰器
import time
def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(f"耗时：{time.time()-start:.2f}秒")