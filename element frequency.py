#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle
from mendeleev import element

# ======================
# 1. 自动获取原子数据
# ======================
def get_atomic_data():
    atomic_data = {}
    for sym in dir(element):
        try:
            el = element(sym)
            if isinstance(el, element.Element):
                atomic_data[el.symbol] = el.atomic_number
        except:
            continue
    return atomic_data

atomic_data = get_atomic_data()

# ======================
# 2. 增强样式配置
# ======================
STYLE_CONFIG = {
    'figure_size': (28, 16),
    'color_map': ['#FFFFFF', '#FF3333'],  # 更鲜艳的红色渐变
    'element': {
        'size': 1.1,
        'border': {
            'color': '#66B2E0',
            'width': 5
        },
        'symbol': {
            'fontsize': 30,
            'active_color': '#2C3E50',    # 有数据的颜色
            'inactive_color': '#BDC3C7',  # 无数据的灰色
            'va': 'center',
            'ha': 'center',
            'y_offset': 0.65
        },
        'count': {
            'fontsize': 24,
            'color': 'black',
            'va': 'top',
            'ha': 'center',
            'y_offset': 0.42
        }
    }
}


# ======================
# 3. 标准周期表坐标定义
# ======================
periodic_table = {
    1: {'H': (0, 0), 'He': (0, 17)},
    2: {
        'Li': (1, 0), 'Be': (1, 1), 
        'B': (1, 12), 'C': (1, 13), 'N': (1, 14), 'O': (1, 15), 'F': (1, 16), 'Ne': (1, 17)
    },
    3: {
        'Na': (2, 0), 'Mg': (2, 1),
        'Al': (2, 12), 'Si': (2, 13), 'P': (2, 14), 'S': (2, 15), 'Cl': (2, 16), 'Ar': (2, 17)
    },
    4: {
        'K': (3, 0), 'Ca': (3, 1),
        'Sc': (3, 2), 'Ti': (3, 3), 'V': (3, 4), 'Cr': (3, 5), 'Mn': (3, 6), 'Fe': (3, 7),
        'Co': (3, 8), 'Ni': (3, 9), 'Cu': (3, 10), 'Zn': (3, 11),
        'Ga': (3, 12), 'Ge': (3, 13), 'As': (3, 14), 'Se': (3, 15), 'Br': (3, 16), 'Kr': (3, 17)
    },
    5: {
        'Rb': (4, 0), 'Sr': (4, 1),
        'Y': (4, 2), 'Zr': (4, 3), 'Nb': (4, 4), 'Mo': (4, 5), 'Tc': (4, 6), 'Ru': (4, 7),
        'Rh': (4, 8), 'Pd': (4, 9), 'Ag': (4, 10), 'Cd': (4, 11),
        'In': (4, 12), 'Sn': (4, 13), 'Sb': (4, 14), 'Te': (4, 15), 'I': (4, 16), 'Xe': (4, 17)
    },
    6: {
        'Cs': (5, 0), 'Ba': (5, 1),
        'La': (5, 2), 'Ce': (8, 3), 'Pr': (8, 4), 'Nd': (8, 5), 'Pm': (8, 6), 'Sm': (8, 7),
        'Eu': (8, 8), 'Gd': (8, 9), 'Tb': (8, 10), 'Dy': (8, 11), 'Ho': (8, 12), 'Er': (8, 13),
        'Tm': (8, 14), 'Yb': (8, 15), 'Lu': (8, 16),
        'Hf': (5, 3), 'Ta': (5, 4), 'W': (5, 5), 'Re': (5, 6), 'Os': (5, 7),
        'Ir': (5, 8), 'Pt': (5, 9), 'Au': (5, 10), 'Hg': (5, 11),
        'Tl': (5, 12), 'Pb': (5, 13), 'Bi': (5, 14), 'Po': (5, 15), 'At': (5, 16), 'Rn': (5, 17)
    },
    7: {
        'Fr': (6, 0), 'Ra': (6, 1),
        'Ac': (6, 2), 'Th': (9, 3), 'Pa': (9, 4), 'U': (9, 5), 'Np': (9, 6), 'Pu': (9, 7),
        'Am': (9, 8), 'Cm': (9, 9), 'Bk': (9, 10), 'Cf': (9, 11), 'Es': (9, 12), 'Fm': (9, 13),
        'Md': (9, 14), 'No': (9, 15), 'Lr': (9, 16),
        'Rf': (6, 3), 'Db': (6, 4), 'Sg': (6, 5), 'Bh': (6, 6), 'Hs': (6, 7),
        'Mt': (6, 8), 'Ds': (6, 9), 'Rg': (6, 10), 'Cn': (6, 11),
        'Nh': (6, 12), 'Fl': (6, 13), 'Mc': (6, 14), 'Lv': (6, 15), 'Ts': (6, 16), 'Og': (6, 17)
    }
}

# ======================
# 4. 增强型绘图函数
# ======================
def create_enhanced_periodic_table():
    plt.figure(figsize=STYLE_CONFIG['figure_size'], facecolor='white', dpi=100)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.axis('off')

    # 创建颜色映射
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'custom_red', STYLE_CONFIG['color_map'])
    
    # 读取数据
    df = pd.read_excel(r"C:\Users\18387\Desktop\element ratio.xlsx")
    element_counts = dict(zip(df['element'], df['times']))
    max_count = df['times'].max()
    norm = mpl.colors.Normalize(vmin=0, vmax=max_count)

    element_size = STYLE_CONFIG['element']['size']
    radius = element_size / 2.25

    for period in periodic_table:
        for elem, (row, col) in periodic_table[period].items():
            count = element_counts.get(elem, 0)
            atomic_number = atomic_data.get(elem, '')
            
            # 计算元素颜色
            base_color = cmap(norm(count)) if count > 0 else 'white'
            
            # 绘制元素圆形
            circle = Circle(
                (col + radius, -row + radius),
                radius=radius,
                facecolor=base_color,
                edgecolor=STYLE_CONFIG['element']['border']['color'],
                linewidth=STYLE_CONFIG['element']['border']['width'],
                zorder=2
            )
            ax.add_patch(circle)

            # 元素符号样式
            symbol_config = STYLE_CONFIG['element']['symbol']
            text_color = (symbol_config['active_color'] 
                         if count > 0 else symbol_config['inactive_color'])
            
            # 绘制元素符号
            plt.text(
                col + radius,
                -row + symbol_config['y_offset'],
                elem,
                fontsize=symbol_config['fontsize'],
                color=text_color,
                weight='bold',
                va=symbol_config['va'],
                ha=symbol_config['ha'],
                zorder=3,
                fontfamily='Arial'
            )

            # 绘制出现次数
            if count > 0:
                count_config = STYLE_CONFIG['element']['count']
                plt.text(
                    col + radius,
                    -row + count_config['y_offset'],
                    str(count),
                    fontsize=count_config['fontsize'],
                    color=count_config['color'],
                    weight='bold',
                    va=count_config['va'],
                    ha=count_config['ha'],
                    zorder=3,
                    fontfamily='Arial'
                )

    # 设置坐标范围
    plt.xlim(-0.5, 18.5)
    plt.ylim(-10.2, 2.5)
    
    # 保存高清图片
    plt.tight_layout()
    save_path = r''
    plt.tight_layout()
    plt.savefig(
        save_path,  # 使用新的保存路径
        dpi=660,
        bbox_inches='tight',
        pad_inches=0.1,
        facecolor='white'
    )
    plt.show()

# ======================
# 5. 执行可视化
# ======================
if __name__ == "__main__":
    create_enhanced_periodic_table()


# In[6]:


import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
import numpy as np

# 设置全局字体为 Arial
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.weight'] = 'bold'

# 创建图形
fig, ax = plt.subplots(figsize=(12, 1.2), dpi=640)
fig.subplots_adjust(bottom=0.5)

# 创建颜色映射
cmap = mpl.colors.LinearSegmentedColormap.from_list('custom_red', ['#FFFFFF', '#FF3333'])
norm = mpl.colors.Normalize(vmin=0, vmax=161)

# 创建颜色条
cb = mpl.colorbar.ColorbarBase(
    ax,
    cmap=cmap,
    norm=norm,
    orientation='horizontal',
    ticks=np.linspace(0, 161, 9)
)

# 设置颜色条边框线宽
cb.outline.set_linewidth(2.5)
cb.outline.set_edgecolor('#66B2E0')
# 设置标签样式
cb.set_label('Element Frequency', fontsize=24, weight='bold', labelpad=10)


# 设置刻度字体加粗（用 FontProperties）
bold_font = font_manager.FontProperties(family='Arial', weight='bold', size=20)
cb.ax.set_xticklabels([str(int(t)) for t in np.linspace(0, 161, 9)], fontproperties=bold_font)

# 保存图像
plt.savefig(r'', dpi=640, bbox_inches='tight', pad_inches=0.1)
plt.show()


# In[ ]:

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import numpy as np
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import os
from pathlib import Path
from collections import Counter

# 设置全局字体
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['font.size'] = 28  # 进一步大幅增大全局字体大小

# 1. 读取数据
file_path = r""
df = pd.read_excel(file_path)
print(f"数据加载完成。总记录数: {len(df)}")
print(f"列名: {df.columns.tolist()}")

# 2. 提取每个单元格的第一个元素
def extract_first_element(element_str):
    """从main element列中提取第一个元素，排除特定元素"""
    if pd.isna(element_str):
        return 'Other'
    
    # 分割元素（按逗号分隔）
    elements = [el.strip() for el in str(element_str).split(',')]

# 提取所有第一个元素用于频率分析
first_elements_list = []
for element_str in df['main element']:
    first_element = extract_first_element(element_str)
    if first_element != 'Other':
        first_elements_list.append(first_element)

# 计算第一个元素的频率
element_counter = Counter(first_elements_list)
print(f"\n第一个元素出现频率:")
for element, count in element_counter.most_common():
    print(f"  {element}: {count}次")

# 只取前30个最频繁的元素
TOP_ELEMENTS = [element for element, count in element_counter.most_common(30)]
print(f"\n前30个最频繁的元素: {TOP_ELEMENTS}")

# 3. 处理main element列 - 只使用第一个元素，且只保留前30个
def extract_main_element(element_str):
    """提取第一个有效元素作为主要元素，只保留前30个最频繁的元素"""
    if pd.isna(element_str):
        return 'Other'
    
    # 分割元素（按逗号分隔）
    elements = [el.strip() for el in str(element_str).split(',')]


# 并行处理main element列
n_jobs = 4
df['Main_Element'] = Parallel(n_jobs=n_jobs)(
    delayed(extract_main_element)(element_str) for element_str in df['main element']
)

# 4. 创建基于元素数量的分组（现在只有单元素）
def create_element_count_groups(df):
    """基于元素数量创建分组 - 现在只有单元素"""
    df['Element_Count'] = 1  # 所有都是单元素
    return df

df = create_element_count_groups(df)
print(f"\n元素数量分组完成:")
print(df['Element_Count'].value_counts().sort_index())

# 5. 创建综合分组（主要元素 + 元素数量）
def create_comprehensive_groups(df):
    """创建综合分组：由于只有单元素，所有都是-Single"""
    groups = []
    for _, row in df.iterrows():
        main_el = row['Main_Element']
        
        if main_el == 'Other':
            groups.append('Other')
        else:
            groups.append(f"{main_el}-Single")  # 所有都是单元素
    
    return groups

df['Comprehensive_Group'] = create_comprehensive_groups(df)
print(f"\n综合分组分布:")
print(df['Comprehensive_Group'].value_counts())

# 确保保存目录存在
save_dir = Path(r"C:\Users\xzr\Desktop\tpeak\plots")
save_dir.mkdir(parents=True, exist_ok=True)

# 定义类别的颜色
CATEGORY_COLORS = {
    ('Main_Element', 'Temperature_peak(℃)'): 'lightcoral',
    ('Main_Element', 'ΔTemperature_peak'): 'royalblue',
    ('Catalyst Categories', 'Temperature_peak(℃)'): '#98FB98',
    ('Catalyst Categories', 'ΔTemperature_peak'): 'orange'
}

# 6. 绘制函数 - 优化版本
def create_advanced_boxplot(df, x_column, y_column, x_label, y_label, plot_type, save_dir):
    """创建高级箱线图 - 修改版本：只显示前30个元素，按频率顺序分批绘制"""
    
    # 获取所有分组（排除Other）
    all_groups = df[x_column].unique()
    all_groups = [g for g in all_groups if pd.notna(g) and g != 'Other']
    
    # 按频率排序分组（使用TOP_ELEMENTS的顺序）
    if 'Main_Element' in x_column:
        # 按元素频率排序
        sorted_groups = []
        for element in TOP_ELEMENTS:
            if element in all_groups:
                sorted_groups.append(element)
        all_groups = sorted_groups
    
    # 每10个元素一批进行绘制
    batch_size = 10
    batches = []
    for i in range(0, len(all_groups), batch_size):
        batch_groups = all_groups[i:i + batch_size]
        batches.append(batch_groups)
    
    # 为每个批次创建图表
    for batch_num, batch_groups in enumerate(batches):
        if len(batch_groups) == 0:
            continue
            
        # 筛选当前批次的数据
        batch_df = df[df[x_column].isin(batch_groups)]
        
        if len(batch_df) == 0:
            continue
            
        # 绘制当前批次的图表
        _create_single_boxplot(batch_df, x_column, y_column, x_label, y_label, 
                              save_dir, batch_num + 1, len(batches), batch_groups)

def _create_single_boxplot(df, x_column, y_column, x_label, y_label, 
                          save_dir, batch_num=1, total_batches=1, current_groups=None):
    """创建单个箱线图"""
    # 大幅增大图形尺寸以适应更大的字体
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 获取当前组合的颜色
    color_key = (x_column, y_column)
    category_color = CATEGORY_COLORS.get(color_key, 'blue')
    
    # 使用传入的分组顺序
    if current_groups is not None:
        x_groups = [g for g in current_groups if g in df[x_column].values]
    else:
        x_groups = df[x_column].unique()
        x_groups = [g for g in x_groups if pd.notna(g) and g != 'Other']
    
    if len(x_groups) == 0:
        print(f"警告: {x_column} 没有有效分组")
        plt.close(fig)
        return
    
    # 设置位置
    positions = range(len(x_groups))
    group_positions = dict(zip(x_groups, positions))
    
    # 准备箱线图数据
    box_data = []
    box_positions = []
    
    for i, group in enumerate(x_groups):
        group_data = df[df[x_column] == group][y_column].dropna()
        if len(group_data) > 0:
            box_data.append(group_data)
            box_positions.append(group_positions[group])
    
    # 绘制箱线图
    if box_data:
        box_plot = ax.boxplot(box_data, positions=box_positions, widths=0.7,
                            patch_artist=True, showfliers=False,
                            boxprops=dict(facecolor='white', alpha=0.9, linewidth=8, edgecolor='black'),
                            whiskerprops=dict(linewidth=8, color='black'),
                            capprops=dict(linewidth=8, color='black'),
                            medianprops=dict(color='red', linewidth=9))
    
    # 绘制散点图（添加jitter）- 使用指定的颜色，无边框
    for i, group in enumerate(x_groups):
        group_data = df[df[x_column] == group][y_column].dropna()
        if len(group_data) > 0:
            jitter = np.random.normal(0, 0.08, len(group_data))
            x_positions = group_positions[group] + jitter
            
            # 使用指定的颜色，移除边框
            ax.scatter(x_positions, group_data, alpha=0.7, s=50,
                      color=category_color, edgecolor='none',
                      zorder=3)
    
    # 修改Y轴标签
    y_label_modified = y_label
    if 'Temperature_peak' in y_label_modified:
        y_label_modified = y_label_modified.replace('Temperature_peak', 'Tp')
    if 'ΔTemperature_peak' in y_label_modified:
        y_label_modified = y_label_modified.replace('ΔTemperature_peak', 'ΔTp')
    
    # 关键修改：为Catalyst Categories设置较小的字体大小
    if x_column == 'Catalyst Categories':
        # Catalyst Categories的所有字体都缩小
        x_label_fontsize = 32  # X轴标签字体
        y_label_fontsize = 32  # Y轴标签字体
        x_tick_fontsize = 24   # X轴刻度字体
        y_tick_fontsize = 30   # Y轴刻度字体
        print(f"设置Catalyst Categories图表字体缩小")
    else:
        # Main_Element保持原来的字体大小
        x_label_fontsize = 42  # X轴标签字体
        y_label_fontsize = 42  # Y轴标签字体
        x_tick_fontsize = 36   # X轴刻度字体
        y_tick_fontsize = 38   # Y轴刻度字体
    
    # 设置轴标签字体
    ax.set_xlabel(x_label, fontsize=x_label_fontsize, weight='bold')
    ax.set_ylabel(y_label_modified, fontsize=y_label_fontsize, weight='bold')
    
    # 设置x轴刻度标签
    ax.set_xticks(positions)
    
    # 简化标签显示
    labels = []
    for group in x_groups:
        labels.append(str(group))
    
    # 设置刻度参数和标签
    ax.tick_params(axis='x', labelsize=x_tick_fontsize, width=6, length=15)
    ax.tick_params(axis='y', labelsize=y_tick_fontsize, width=6, length=15)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # 设置边框粗细
    for spine in ax.spines.values():
        spine.set_linewidth(6)
    
    # 添加网格
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=3)
    
    plt.tight_layout()
    
    # 修改保存文件名中的Y轴标签
    y_column_modified = y_column
    if 'Temperature_peak' in y_column_modified:
        y_column_modified = y_column_modified.replace('Temperature_peak', 'Tp')
    if 'ΔTemperature_peak' in y_column_modified:
        y_column_modified = y_column_modified.replace('ΔTemperature_peak', 'ΔTp')
    
    # 保存图像
    base_filename = f"boxplot_{y_column_modified.replace('(', '').replace(')', '').replace('/', 'per').replace('℃', 'C')}_by_{x_column}"
    if total_batches > 1:
        filename = f"{base_filename}_part{batch_num}of{total_batches}.png"
    else:
        filename = f"{base_filename}.png"
    
    save_path = save_dir / filename
    
    try:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"图像已保存至: {save_path}")
    except Exception as e:
        print(f"保存图像时出错: {e}")
    
    plt.show()
    plt.close(fig)

# 7. 绘制多个不同的箱线图
print("\n开始绘制箱线图...")

# 修改Y轴标签
y_label_mapping = {
    'Temperature_peak(℃)': 'Tp(°C)',
    'ΔTemperature_peak': 'ΔTp(°C)'
}

# 定义要绘制的组合（移除了Comprehensive_Group相关的图）
plot_combinations = [
    # (x_column, y_column, x_label, y_label, plot_type)
    ('Main_Element', 'Temperature_peak(℃)', 'Main Element', y_label_mapping['Temperature_peak(℃)'], 'main_element'),
    ('Main_Element', 'ΔTemperature_peak', 'Main Element', y_label_mapping['ΔTemperature_peak'], 'main_element'),
    ('Catalyst Categories', 'Temperature_peak(℃)', 'Catalyst Categories', y_label_mapping['Temperature_peak(℃)'], 'catalyst'),
    ('Catalyst Categories', 'ΔTemperature_peak', 'Catalyst Categories', y_label_mapping['ΔTemperature_peak'], 'catalyst'),
]

for x_col, y_col, x_label, y_label, plot_type in plot_combinations:
    if y_col in df.columns:
        print(f"绘制 {x_label} + {y_label}...")
        print(f"使用颜色: {CATEGORY_COLORS.get((x_col, y_col), '默认颜色')}")
        create_advanced_boxplot(df, x_col, y_col, x_label, y_label, plot_type, save_dir)
    else:
        print(f"未找到 {y_col} 列")

print("\n=== 最终数据统计 ===")
print(f"总数据点数: {len(df)}")
print(f"\n主要元素分布（前30个元素）:")
main_element_counts = df['Main_Element'].value_counts()
# 只显示前30个元素
top_30_elements = main_element_counts[main_element_counts.index.isin(TOP_ELEMENTS)]
print(top_30_elements)
print(f"\n元素数量分布（现在都是单元素）:")
print(df['Element_Count'].value_counts().sort_index())
print(f"\n催化剂分类分布:")
print(df['Catalyst Categories'].value_counts())

# 数据完整性检查
print(f"\n=== 数据完整性检查 ===")
if 'Temperature_peak(℃)' in df.columns:
    temp_data = df['Temperature_peak(℃)'].dropna()
    print(f"Temperature_peak(℃) 非空值: {len(temp_data)}")
    if len(temp_data) > 0:
        print(f"Temperature_peak(℃) 范围: {temp_data.min():.1f} - {temp_data.max():.1f}")

if 'ΔTemperature_peak' in df.columns:
    delta_data = df['ΔTemperature_peak'].dropna()
    print(f"ΔTemperature_peak 非空值: {len(delta_data)}")
    if len(delta_data) > 0:
        print(f"ΔTemperature_peak 范围: {delta_data.min():.1f} - {delta_data.max():.1f}")

print("\n所有图表绘制完成！")


