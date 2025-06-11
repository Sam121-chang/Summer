"""
封装是将数据和操作数据的方法绑定在一起，通过访问修饰符（如_或__)控制属性的可见性。
把数据和方法装在类里，像把玩具放进盒子，只通过方法（盒子的按钮）操作数据
"""
#创建一个银行账户类
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

account = BankAccount('001','张三',1000)
print(account.owner)
print(account.get_balance())
account.deposit(500)
print(account.get_balance())
#print(account._account)  会报错，无法直接访问私有属性
"""
关键点：
_account_id:保护属性，约定不直接访问（但Python不强制）
__balance:私有属性，外部无法直接访问，需通过get_balance()方法获取
deposit和withdraw:封装了余额修改逻辑，确保数据安全
"""