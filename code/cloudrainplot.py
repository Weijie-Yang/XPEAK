#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ptitprince as pt
import numpy as np

# 设置全局字体和样式
plt.rcParams['font.family'] = 'Arial'  # 全局字体设置为Arial
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
sns.set(style="whitegrid")  # 设置Seaborn样式

# 数据准备 -------------------------------------------------------------------
file_path = r''  # 修改为实际路径
data = pd.read_excel(file_path)

# 数据清洗和重组
raincloud_data = pd.concat([
    data[['Temperature_peak(℃)']]
        .dropna()
        .rename(columns={'Temperature_peak(℃)': 'Value'})
        .assign(Category='Tp(°C)'),
    
    data[['\u0394Temperature_peak']]
        .dropna()
        .rename(columns={'\u0394Temperature_peak': 'Value'})
        .assign(Category='ΔTp(°C)')
])

# 可视化设置 -------------------------------------------------------------------
plt.figure(figsize=(10, 6),dpi=640)

# 绘制雨云图
pt.RainCloud(
    x='Category',
    y='Value',
    data=raincloud_data,
    palette=["#FF9999", "#89CFF0"],
    bw=.2,
    width_viol=.6,
    width_box=.1,
    orient="v",
    alpha=.8,
    move=-0.0
)

# 获取当前坐标轴
ax = plt.gca()

# 轴标签设置
ax.set_xlabel("", fontsize=30, fontweight='bold')
ax.set_ylabel("Temperature (°C)", fontsize=26, fontweight='bold')

# 刻度设置
ax.tick_params(axis='both', labelsize=22, width=2)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

# 统计标注 -------------------------------------------------------------------
for i, category in enumerate(['Tp(°C)', 'ΔTp(°C)']):
    category_data = raincloud_data[raincloud_data['Category'] == category]['Value'].dropna()

    if category_data.empty:
        continue  # 跳过无数据的类别

    # 计算统计量
    q25 = category_data.quantile(0.25)
    q75 = category_data.quantile(0.75)
    median = category_data.median()
    iqr = q75 - q25
    upper = min(category_data.max(), q75 + 1.5 * iqr)
    lower = max(category_data.min(), q25 - 1.5 * iqr)

    # 确保所有数值都是有限的
    stats = [median, q25, q75, lower, upper]
    if not all(np.isfinite(stats)):
        continue

    # 标注位置参数
    x_offset = i + 0.44  # 向右偏移量
    text_params = {
        'ha': 'center',
        'va': 'center',
        'fontsize': 18,
        'fontweight': 'bold'
    }

    # 添加统计标注
    ax.text(x_offset, median, f'Median: {median:.2f}°C', color='black', **text_params)
    ax.text(x_offset, q25, f'Q1: {q25:.2f}°C', color='black', **text_params)
    ax.text(x_offset, q75, f'Q3: {q75:.2f}°C', color='black', **text_params)
    ax.text(x_offset, lower, f'Lower: {lower:.2f}°C', color='#6495ED', **text_params)
    ax.text(x_offset, upper, f'Upper: {upper:.2f}°C', color='#CC6666', **text_params)

# 边框样式调整
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(2.5)

# x轴范围微调
x_lim = ax.get_xlim()
ax.set_xlim(x_lim[0], x_lim[1] + 0.44)

# 保存和显示 ------------------------------------------------------------------
output_path = r'C:\Users\xzr\Desktop\paperpic\RainCloud_Plot.png'
plt.savefig(output_path, dpi=640, bbox_inches='tight', format='png')
plt.show()


# In[ ]:




