#数据准备与处理
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 生成模拟股票数据
dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='B')
np.random.seed(42)
prices = np.random.randn(len(dates)).cumsum() + 100
volume = np.random.randint(1000000, 10000000, size=len(dates))

df = pd.DataFrame({
    'date': dates,
    'price': prices,
    'volume': volume
})

# 计算移动平均线
df['MA_5'] = df['price'].rolling(5).mean()
df['MA_20'] = df['price'].rolling(20).mean()

#多面板可视化
# 创建画布与双Y轴
fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()  # 共享x轴

# 绘制价格与均线
ax1.plot(df['date'], df['price'], 'b-', label='Price')
ax1.plot(df['date'], df['MA_5'], 'r--', label='5-Day MA')
ax1.plot(df['date'], df['MA_20'], 'g--', label='20-Day MA')
ax1.set_ylabel('Price (USD)', color='b')

# 绘制成交量
ax2.bar(df['date'], df['volume'], color='gray', alpha=0.3, label='Volume')
ax2.set_ylabel('Volume', color='gray')

# 添加标题与图例
plt.title('Stock Price and Volume')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# 设置x轴日期格式
fig.autofmt_xdate()

plt.show()