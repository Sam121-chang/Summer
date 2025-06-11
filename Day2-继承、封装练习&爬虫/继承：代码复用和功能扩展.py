"""
继承允许子类继承父类的属性和方法，并可以重写和扩展它们

"""
class BankAccount:
    def __init__(self,account_id,owner,balance=0):
        self._account_id = account_id  #保护属性，单下划线
        self.__balance=balance         #私有属性，双下划线
        self.owner=owner               #公有属性

    def deposit(self,amount):
        if amount > 0:
            self.__balance += amount
            return True
        return False

    def withdraw(self,amount):
        if 0 < amount <= self._balance:
            self.__balance -= amount
            return True
        return False

    def get_balance(self):
        return self.__balance


class SavingsAccount(BankAccount):
    def __init__(self, account_id, owner, balance=0, interest_rate=0.02):
        super().__init__(account_id, owner, balance)  # 调用父类构造函数
        self.interest_rate = interest_rate

    def add_interest(self):
        """计算并添加利息"""
        interest = self.get_balance() * self.interest_rate
        self.deposit(interest)
        return interest


class CheckingAccount(BankAccount):
    def __init__(self, account_id, owner, balance=0, overdraft_limit=1000):
        super().__init__(account_id, owner, balance)
        self.__overdraft_limit = overdraft_limit

    def withdraw(self, amount):  # 重写父类方法
        """支持透支的取款操作"""
        if 0 < amount <= self.get_balance() + self.__overdraft_limit:
            self._BankAccount__balance -= amount  # 直接访问父类私有属性（不推荐，但可行）
            return True
        return False

savings = SavingsAccount('S001','李四',2000)
print(f'利息：{savings.add_interest()}')

checking = CheckingAccount('C001','王五',500,1000)
checking.withdraw(1200)
print(checking.get_balance())

"""
super().__init__():调用父类构造函数，初始化继承的属性
add_interest():子类特有方法，扩展父类功能
重写withdraw():子类修改父类行为，支持透支
"""



#多态：统一接口，不同实现
#多态允许不同类的对象对同一方法做出不同相应
class Shape:
    def area(self):
        pass

class Rectangle(Shape):
    def __init__(self,width,height):
        self.width = width
        self.height = height

    def area(self): #实现父类抽象方法
        return self.width * self.height

class Circle(Shape):
    def __inir__(self,radius):
        self.radius = radius

    def area(self):
        return 3.14 * self.radius **2

#多态应用
def print_area(shape):
    print(f"面积：{shape.area()}")

rect = Rectangle(5,10)
circle = Circle(3)

print_area(rect)
print_area(circle)
"""
print_area()函数接受任何Shape子类的对象，自动调用其area()方法；
子类必须实现父类的抽象方法（如area()），否则会报错
"""