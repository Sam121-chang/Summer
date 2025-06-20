#1. 大数据可视化技巧
# 对于百万级数据点，使用agg渲染器
import matplotlib as mpl
import numpy as np
import mpld3
import matplotlib.pyplot as plt
mpl.use('Agg')  # 在脚本开头设置


# 采样绘制大数据
def plot_large_data(data, sample_size=1000):
    if len(data) > sample_size:
        indices = np.linspace(0, len(data) - 1, sample_size, dtype=int)
        sampled_data = data.iloc[indices]
    else:
        sampled_data = data

    plt.plot(sampled_data.index, sampled_data['value'])
    plt.show()


# 交互式图表生成
# 生成HTML交互式图表
import numpy as np
import matplotlib.pyplot as plt
import mpld3

# 定义 x 变量
x = np.linspace(0, 10, 100)  # 创建从 0 到 10 的 100 个点

fig, ax = plt.subplots()
ax.plot(x, np.sin(x))
mpld3.save_html(fig, 'sin_wave.html')
print("交互式图表已保存为 sin_wave.html")

fig, ax = plt.subplots()
ax.plot(x, np.sin(x))
mpld3.save_html(fig, 'sin_wave.html')  # 保存为HTML文件