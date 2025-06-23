"""项目要求：处理一个包含用户信息的JSON文件，提取关键信息，生成统计报告并可视化"""
import json
import pandas as pd
import matplotlib.pyplot as plt

# 1. 读取JSON数据
with open('users.json', 'r') as f:
    users = json.load(f)

# 2. 提取关键信息到DataFrame
data = []
for user in users:
    data.append({
        'id': user['id'],
        'name': user['name'],
        'age': user['age'],
        'city': user['location']['city'],
        'signup_date': user['signup_date'],
        'total_purchase': sum(order['amount'] for order in user['orders'])
    })

df = pd.DataFrame(data)

# 3. 数据处理
# 转换日期格式
df['signup_date'] = pd.to_datetime(df['signup_date'])

# 计算每月注册用户数
monthly_signups = df['signup_date'].dt.to_period('M').value_counts().sort_index()

# 按城市分组统计
city_stats = df.groupby('city').agg({
    'id': 'count',
    'age': 'mean',
    'total_purchase': 'sum'
}).rename(columns={'id': 'user_count', 'age': 'avg_age', 'total_purchase': 'total_sales'})

# 4. 生成报告
print("用户统计报告：")
print(f"总用户数：{len(df)}")
print(f"平均年龄：{df['age'].mean():.1f}")
print(f"总消费金额：{df['total_purchase'].sum()}")
print("\n各城市统计：")
print(city_stats)

# 5. 可视化
plt.figure(figsize=(12, 5))

# 绘制每月注册用户数
plt.subplot(1, 2, 1)
monthly_signups.plot(kind='bar')
plt.title('每月注册用户数')
plt.xlabel('月份')
plt.ylabel('用户数')

# 绘制城市总销售额
plt.subplot(1, 2, 2)
city_stats['total_sales'].plot(kind='bar')
plt.title('各城市总销售额')
plt.xlabel('城市')
plt.ylabel('销售额')

plt.tight_layout()
plt.savefig('user_stats.png')
plt.show()

# 6. 保存处理后的数据
df.to_csv('processed_users.csv', index=False)
with open('city_stats.json', 'w') as f:
    json.dump(city_stats.to_dict(), f, indent=2)