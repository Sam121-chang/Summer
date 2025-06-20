"""
一、Matplotlib 核心架构：三层 API 与图形组件
（一）三层 API 体系
底层（Backend）：控制图形渲染（如 Agg、TkAgg）；
中层（Artist）：处理所有可视元素（Figure、Axes 等）；
顶层（pyplot）：面向用户的命令式接口。

（二）图形组件关系
plaintext
Figure（画布） → Axes（坐标系） → Axis（坐标轴） → Artist（元素）

关键对象：
Figure：顶级容器，可包含多个 Axes；
Axes：实际绘图区域，包含两个 Axis（x 轴和 y 轴）。

"""

#基础绘图

#单图
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,10,100)
y = np.sin(x)

fig,ax = plt.subplots()

ax.plot(x,y,label='sin(x)')

ax.set_title('正弦函数')
ax.set_xlabel('x')
ax.set_ylabel('y')

ax.legend()

plt.show()


#多子图布局
#1.使用subplots()创建网络
fig,axes = plt.subplots(2,2,figsize=(10,8))

axes[0,0].plot(x,np.sin(x))
axes[0,1].plot(x,np.cos(x))
axes[1,0].plot(x,np.tan(x))
axes[1,1].plot(x,x**2)

for i in range(2):
    for j in range(2):
        axes[i,j].set_title(f"子图({i},{j})")
plt.tight_layout() #自动调整布局
plt.show()

#2.使用GridSpec自定义布局
from matplotlib.gridspec import GridSpec

fig=plt.figure(figsize=(10,8))
gs = GridSpec(2,2,figure=fig)

ax1 = fig.add_subplot(gs[0,:])
ax1.plot(x,np.cos(x))

ax2 = fig.add_subplot(gs[1,1])
ax2.plot(x,np.cos(x))

ax3 = fig.add_subplot(gs[1,1])
ax3.plot(x,np.tan(x))

plt.show()


"""
高级绘图技巧：自定义与美化
"""
#曲线样式定制
fig, ax = plt.subplots()

# 绘制多条曲线，设置不同样式
ax.plot(x, np.sin(x), 'r-', linewidth=2, label='sin(x)')  # 红色实线
ax.plot(x, np.cos(x), 'b--', alpha=0.7, label='cos(x)')  # 蓝色虚线，70%透明度
ax.plot(x, x/5, 'g.', markersize=10, label='x/5')  # 绿色点

# 添加注释
ax.annotate('最大值', xy=(np.pi/2, 1), xytext=(2, 0.8),
           arrowprops=dict(facecolor='black', shrink=0.05))

plt.legend()
plt.show()


#色彩映射与填充
# 生成二维数据
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

# 绘制热力图
fig, ax = plt.subplots()
im = ax.imshow(Z, cmap='viridis', origin='lower', extent=[-5, 5, -5, 5])
fig.colorbar(im)  # 添加颜色条

# 绘制等高线
ax.contour(X, Y, Z, 10, colors='black', alpha=0.5)

plt.show()
























