"""
类 Class是对一类事物的抽象描述，定义了事物的属性（数据）和方法（行为）；
对象Object是类的具体实例
"""
#定义一个动物类
class Animal:
    def __init__(self,name,age): #__init__是初始化方法，self代表对象本身
        self.name = name #属性：名字
        self.age = age #属性：年龄

    def speak(self):
        raise NotImplementedError("子类需要实现speak方法")

    def show_info(self):
        return f"名称:{self.name},年龄:{self.age}"



"""
继承Inheritance：代码复用的核心
子类可以继承父类的属性和方法，避免重复编写代码
"""
class Dog(Animal):
    def __init__(self,name,age,breed):
        super().__init__(name,age)
        self.breed = breed

    def speak(self):
        return"汪汪汪！"

#创建狗对象
my_dog = Dog("旺财",2,"金毛")
print(my_dog.speak())
print(my_dog.show_info())

"""
多态Polymorphism：同一方法的不同实现
"""
class Cat(Animal):
    def speak(self):
        return"喵喵喵"

#多态应用：统一接口处理不同对象
def make_animal_speak(animal):
    print(animal.speak())

make_animal_speak(Cat("来福",1))
make_animal_speak(my_dog)