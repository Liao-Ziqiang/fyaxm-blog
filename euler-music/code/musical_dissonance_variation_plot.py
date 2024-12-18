import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from matplotlib.cm import get_cmap

# 原始数据
data = [
    10, 15, 16, 10, 10, 16, 11, 12,
    12, 20, 10, 26, 10, 26, 11, 12,
    12, 20, 10, 18, 10, 20, 30, 20,
    20, 10, 13, 20, 24, 10, 13, 20,
    18, 17, 18, 10
]

# 横坐标序号（从1开始）
original_x = np.arange(1, len(data) + 1)

# 样条插值：在每两个相邻点之间插入更多插值点
new_x = np.linspace(original_x[0], original_x[-1], len(data) * 5)  # 每两个点插值20个
spline = make_interp_spline(original_x, data, k=3)  # 三次样条插值
new_data = spline(new_x)  # 生成插值后的数据

# 伪彩色映射
norm = plt.Normalize(min(new_data - 5), max(new_data + 5))  # 根据数据范围归一化
colors = get_cmap("viridis")(norm(new_data))  # 使用伪彩色映射

# 设置图形大小和风格
plt.figure(figsize=(18, 4))  # 长条状
plt.style.use('ggplot')

# 绘制柱状图
# 注意调整柱宽，让柱子完全紧贴在一起，并去掉边缘
bar_width = (new_x[1] - new_x[0])  # 柱宽完全覆盖间距
bars = plt.bar(new_x, new_data, color=colors, width=bar_width, linewidth=0)  # 设置linewidth=0移除边缘

# 添加颜色条（Colorbar）
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, aspect=30, pad=0.02)
cbar.set_label('Value', fontsize=12)

# 添加标题和标签
plt.title("Smooth Bar Chart with Cubic Spline Interpolation (No Gaps)", fontsize=14)
plt.xlabel("Index", fontsize=12)
plt.ylabel("Value", fontsize=12)

# 调整横轴刻度：只显示原始横坐标的序号
plt.xticks(ticks=original_x, labels=original_x, fontsize=10, rotation=45)

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()