#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


# In[2]:


from sklearn.decomposition import PCA  # 修改导入PCA
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_excel(r"C:\Users\18387\Desktop\train_1.xlsx")
y = df['Temperature_peak(℃)'].values
excluded_columns = [
    "paper_number", "catalyst_number", "catalysts_component",
    "Temperature_peak(℃)", "ΔTemperature_peak"
]
X = df.drop(excluded_columns, axis=1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 定义需要保留的列（不参与降维）
exclude_for_pca = ["Catalysts_Mass_Fraction(wt%)", "heating_rate(℃/min)"]

# 提取保留列
X_train_retained = X_train[exclude_for_pca].values
X_test_retained = X_test[exclude_for_pca].values

# 准备降维数据
X_train_for_pca = X_train.drop(exclude_for_pca, axis=1)
X_test_for_pca = X_test.drop(exclude_for_pca, axis=1)

# 使用PCA降维到60维
pca = PCA(n_components=90, random_state=42)
X_train_pca = pca.fit_transform(X_train_for_pca)
X_test_pca = pca.transform(X_test_for_pca)

# 合并保留列与PCA特征
X_train_combined = np.hstack([X_train_retained, X_train_pca])
X_test_combined = np.hstack([X_test_retained, X_test_pca])

print("训练集合并后维度:", X_train_combined.shape)
print("测试集合并后维度:", X_test_combined.shape)   # 同上


# In[3]:


rf = RandomForestRegressor(n_estimators=550, random_state=42,max_depth=None,min_samples_split=3,min_samples_leaf=2,criterion='absolute_error',
                           min_weight_fraction_leaf=0.0, max_features='log2', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, 
                           oob_score=False, n_jobs=-1,verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None)

rf.fit(X_train_combined, y_train)

# 训练集评估（使用X_train_combined）
train_pred = rf.predict(X_train_combined)
print('Training R2 = %.3f' % r2_score(y_train, train_pred))
print('Training RMSE = %.3f' % np.sqrt(mean_squared_error(y_train, train_pred)))
print('Training MAE = %.3f' % mean_absolute_error(y_train, train_pred))

# 测试集评估（使用X_test_combined）
test_pred = rf.predict(X_test_combined)
print('\nTesting R2 = %.3f' % r2_score(y_test, test_pred))
print('Testing RMSE = %.3f' % np.sqrt(mean_squared_error(y_test, test_pred)))
print('Testing MAE = %.3f' % mean_absolute_error(y_test, test_pred))

# 交叉验证（使用X_train_combined）
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(rf, X_train_combined, y_train, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=-1)
rmse_scores = [np.sqrt(abs(s)) for s in scores]
r2_scores = cross_val_score(rf, X_train_combined, y_train, scoring='r2', cv=crossvalidation, n_jobs=-1)
mae_scores = cross_val_score(rf, X_train_combined, y_train, scoring='neg_mean_absolute_error', cv=crossvalidation, n_jobs=-1)

print('\nCross-validation results:')
print('Folds: %i, mean R2: %.3f' % (len(r2_scores), np.mean(r2_scores)))
print('Folds: %i, mean RMSE: %.3f' % (len(rmse_scores), np.mean(rmse_scores)))
print('Folds: %i, mean MAE: %.3f' % (len(mae_scores), np.mean(np.abs(mae_scores))))


# In[4]:


import shap
import matplotlib.pyplot as plt
import numpy as np


# === 特征名修正 ===
modified_names = exclude_for_pca.copy()

# 使用正确的原始特征名进行替换
modified_names = [name.replace('heating_rate(℃/min)', 'Heating Rate') 
                 for name in modified_names]
                 
modified_names = [name.replace('Catalysts_Mass_Fraction(wt%)', 'Cat. Fraction') 
                 for name in modified_names]

# 更新保留的特征名称
retained_feature_names = modified_names

# NMF特征名
pca_feature_names = [f'PCA_{i}' for i in range(X_train_pca.shape[1])]

# 合并所有特征名
all_feature_names = retained_feature_names + pca_feature_names

# 打印验证
print("修正后的特征名称:", all_feature_names[:5])  # 打印前5个验证



# === 绘图样式设置 ===
plt.rcParams.update({
    'figure.facecolor': (0, 0, 0, 0),  # 透明背景
    'axes.facecolor': (0, 0, 0, 0),   # 透明背景
    'axes.linewidth': 2.5,             # 加粗坐标轴线
    'xtick.major.width': 2,            # 加粗x轴刻度线
    'ytick.major.width': 2,            # 加粗y轴刻度线
    'xtick.major.size': 14,             # 加大刻度尺寸
    'ytick.major.size': 14,             # 加大刻度尺寸
    'font.weight': 'bold',             # 加粗字体
    'axes.labelweight': 'bold',        # 加粗轴标签
    'axes.titleweight': 'bold'         # 加粗标题
})

# --- 创建SHAP解释器 ---
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_train_combined)

# ===== 特征重要性图 - 修正顺序问题 =====
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_train_combined, feature_names=all_feature_names, 
                  plot_type="bar", max_display=10, show=False)

# 获取当前坐标轴并设置样式
ax = plt.gca()

# 计算特征重要性值（平均绝对SHAP值）
feature_importance = np.mean(np.abs(shap_values), axis=0)
# 获取前10个最重要的特征索引（按降序排列）
top_indices = np.argsort(feature_importance)[::-1][:10]
 

# 获取所有条形对象（从顶部到底部）
bars = ax.containers[0]

for bar in ax.containers[0]:
    bar.set_color('skyblue')  
    bar.set_alpha(0.6)  # 添加透明度

# 添加重要性数值标签（按从上到下的顺序）
for i in range(len(bars)):
    bar = bars[i]
    # 获取对应特征的重要性值（按降序索引）
    value = feature_importance[top_indices[len(bars)-1-i]]
    
    # 在条形内部左侧添加数值标签
    # 偏移量为条形宽度的1%，确保在条内左侧
    text_x = bar.get_width() * 0.02
    text_y = bar.get_y() + bar.get_height()/2
    
    # 格式化数值（保留两位小数）
    ax.text(text_x, text_y, f'{value:.1f}', 
            fontsize=14, fontweight='bold', color='black',
            verticalalignment='center')

# 将整个横坐标轴（刻度和边框）移动到上方
ax.xaxis.set_ticks_position('top')       # 刻度移动到顶部
ax.xaxis.set_label_position('top')       # 标签移动到顶部
ax.spines['top'].set_position(('axes', 1.0))  # 顶边框保持原位

ax.set_xlabel('Feature Importance', fontsize=20, fontweight='bold', labelpad=10)

# 将横坐标轴线移动到上方位置
ax.spines['bottom'].set_visible(False)   # 隐藏底部轴线
ax.spines['top'].set_visible(True)       # 确保顶部轴线可见
ax.spines['top'].set_linewidth(2.5)       # 加粗顶部轴线
ax.spines['top'].set_zorder(10)          # 确保轴线在最上层

# 调整纵轴位置到左侧
ax.spines['left'].set_position(('axes', 0.0))
ax.yaxis.set_ticks_position('left')
ax.yaxis.set_label_position('left')

# 隐藏右侧和顶部的多余轴线
ax.spines['right'].set_visible(False)

# 设置透明背景和边框线宽
ax.set_facecolor((0, 0, 0, 0))
for spine in ax.spines.values():
    spine.set_linewidth(2.5)

# === 加大刻度尺寸 ===
# 增加刻度线的长度和宽度
ax.tick_params(axis='x', which='major', width=3, size=14)  # x轴刻度（顶部）
ax.tick_params(axis='y', which='major', width=3, size=14)  # y轴刻度（左侧）

# 增加刻度标签的字体大小
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)

plt.title("", fontsize=14, fontweight='bold')
plt.tight_layout()

# ===== 详细特征影响点图 - 只显示前10个 =====
plt.figure(figsize=(12, 8), dpi=100)  # 增加DPI以获得更清晰的渲染
# 绘制SHAP点图，但不显示（show=False）
shap.summary_plot(
    shap_values, 
    X_train_combined, 
    feature_names=all_feature_names, 
    max_display=10, 
    show=False
)

# 获取当前坐标轴
ax2 = plt.gca()

# 设置横坐标轴名称
ax2.set_xlabel('SHAP Value', fontsize=20, fontweight='bold', labelpad=10)

# 双重保障设置标签
if ax2.get_xlabel() == '':
    ax2.set_xlabel('SHAP Value', fontsize=20, fontweight='bold', labelpad=10)

# 加大刻度尺寸
ax2.tick_params(axis='x', which='major', width=3, size=14)
ax2.tick_params(axis='y', which='major', width=3, size=14)
ax2.tick_params(axis='x', labelsize=16)
ax2.tick_params(axis='y', labelsize=16)

# === 可靠方法加大颜色条字体 ===
# 1. 获取所有axes
all_axes = plt.gcf().get_axes()

# 2. 查找颜色条对象
cbar = None
for ax in all_axes:
    # 尝试获取颜色条对象
    if hasattr(ax, 'collections') and len(ax.collections) > 0:
        # 查找散点图集合
        for collection in ax.collections:
            if hasattr(collection, 'colorbar'):
                cbar = collection.colorbar
                break
    if cbar is not None:
        break

# 3. 如果找到颜色条，直接修改其属性
if cbar is not None:
    # 方法1：直接修改颜色条刻度标签
    cbar.ax.tick_params(labelsize=16)
    
    # 方法2：重新设置颜色条标签
    cbar.set_label(cbar.ax.get_ylabel(), size=16, weight='bold')
    
    # 方法3：强制更新颜色条
    cbar.update_normal(cbar.mappable)

# 4. 如果上述方法失败，使用备用方法
if cbar is None:
    # 查找所有颜色条axes
    for ax in all_axes:
        if 'colorbar' in ax.get_label().lower():
            # 加大颜色条刻度标签字体
            ax.tick_params(labelsize=16)
            
            # 加大颜色条标题字体
            cbar_label = ax.get_ylabel()
            ax.set_ylabel(cbar_label, fontsize=16, fontweight='bold')
            
            # 强制重绘
            ax.figure.canvas.draw_idle()
            break

# 确保标题为空
plt.title("", fontsize=14, fontweight='bold')

# 手动调整布局
plt.tight_layout(rect=[0, 0, 1, 0.97])  # 为顶部标签留出空间

# 显式刷新图形 - 使用更强大的方法
plt.gcf().canvas.draw()
plt.gcf().canvas.flush_events()

# ===== 单样本解释 ===== 
plt.figure(figsize=(12, 4))
shap.force_plot(
    explainer.expected_value,
    shap_values[0, :],
    X_train_combined[0, :],
    feature_names=all_feature_names,
    matplotlib=True,
    text_rotation=15,
    show=False
)
plt.title("Individual Prediction Explanation", fontsize=14, fontweight='bold')
ax = plt.gca()
ax.set_facecolor((0, 0, 0, 0))
# 确保所有四个边框都可见并加粗
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(2.5)
plt.tight_layout()

# ===== 交互作用分析 - 双视角展示 =====
catalyst_idx = all_feature_names.index('Cat. Fraction')
heating_idx = all_feature_names.index('Heating Rate')

# 视角1：以催化剂负载量为主，加热速率为交互项
fig1, ax1 = plt.subplots(figsize=(10, 6))

# 首先绘制依赖图以获取数据范围
shap.dependence_plot(
    catalyst_idx,
    shap_values,
    X_train_combined,
    feature_names=all_feature_names,
    interaction_index=heating_idx,
    alpha=0.7,
    dot_size=16,
    show=False,
    ax=ax1
)

# 获取数据范围（包括padding）
xmin_data, xmax_data = ax1.get_xlim()
ymin_data, ymax_data = ax1.get_ylim()

# 计算扩展范围（消除边距）
padding_factor = 0.02  # 无padding
xmin = xmin_data - padding_factor * (xmax_data - xmin_data)
xmax = xmax_data + padding_factor * (xmax_data - xmin_data)
ymin = ymin_data - padding_factor * (ymax_data - ymin_data)
ymax = ymax_data + padding_factor * (ymax_data - ymin_data)

# 清除当前图形
ax1.clear()

# 设置背景分区 - 使用扩展的范围确保完全覆盖
# 小于5的区域 - 淡灰色
ax1.axvspan(xmin=xmin, xmax=4.8, color='lightgray', alpha=0.45, zorder=0)
# 5-10的区域 - 淡红色
ax1.axvspan(xmin=5, xmax=9.8, color='lightblue', alpha=0.35, zorder=0)
# 大于10的区域 - 淡蓝色
ax1.axvspan(xmin=10, xmax=xmax, color='lightcoral', alpha=0.3, zorder=0)

# 添加SHAP=0的黑色虚线，加粗
ax1.axhline(y=0, color='black', linestyle='--', linewidth=2.5, zorder=1)

# 重新绘制依赖图
shap.dependence_plot(
    catalyst_idx,
    shap_values,
    X_train_combined,
    feature_names=all_feature_names,
    interaction_index=heating_idx,
    alpha=0.7,
    dot_size=16,
    show=False,
    ax=ax1
)

# 设置固定范围防止自动调整留白
ax1.set_xlim(xmin, xmax)
ax1.set_ylim(ymin, ymax)

# 美化设置
ax1.set_title("", fontsize=14, fontweight='bold')  # 标题为空白

# 确保所有四个边框都可见并加粗
for spine in ax1.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(2.5)
    
ax1.set_facecolor((1, 1, 1, 0.2))

# 加大坐标轴数字大小和刻度线尺寸
ax1.tick_params(axis='both', width=2.5, length=8, labelsize=18)

# 设置X轴和Y轴标签
ax1.set_xlabel("Catalyst Mass Fraction", fontsize=24, fontweight='bold')
ax1.set_ylabel("SHAP Value", fontsize=24, fontweight='bold')

# 找到并修改颜色条对象
cbar = [child for child in ax1.get_children() if isinstance(child, plt.cm.ScalarMappable)]
if cbar:
    cbar = cbar[0]
    # 获取颜色条对应的坐标轴
    cax = cbar.colorbar.ax
    # 设置颜色条标签
    cax.set_ylabel('Heating Rate', fontsize=22, fontweight='bold', labelpad=20)
    # 设置颜色条刻度大小
    cax.tick_params(labelsize=16, length=5, width=2)
    # 加粗颜色条边框
    for spine in cax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0)
        
plt.tight_layout()


# 视角2：以加热速率为主，催化剂负载量为交互项
fig2, ax2 = plt.subplots(figsize=(10, 6))

# 首先绘制依赖图获取范围
shap.dependence_plot(
    heating_idx,
    shap_values,
    X_train_combined,
    feature_names=all_feature_names,
    interaction_index=catalyst_idx,
    alpha=0.7,
    dot_size=16,
    show=False,
    ax=ax2
)

# 获取数据范围（包括padding）
xmin_data, xmax_data = ax2.get_xlim()
ymin_data, ymax_data = ax2.get_ylim()

# 计算扩展范围（消除边距）
padding_factor = 0.2
xmin = xmin_data - padding_factor * (xmax_data - xmin_data)
xmax = xmax_data + padding_factor * (xmax_data - xmin_data)
ymin = ymin_data - padding_factor * (ymax_data - ymin_data)
ymax = ymax_data + padding_factor * (ymax_data - ymin_data)

# 清除当前图形
ax2.clear()

# 设置背景分区 - 以10为分界
# 加热速率小于10的区域 - 淡蓝色
ax2.axvspan(xmin=xmin, xmax=10, color='lightblue', alpha=0.35, zorder=0)
# 加热速率大于等于10的区域 - 淡红色
ax2.axvspan(xmin=10, xmax=xmax, color='lightcoral', alpha=0.3, zorder=0)

# 添加SHAP=0的黑色虚线，加粗
ax2.axhline(y=0, color='black', linestyle='--', linewidth=2.5, zorder=1)

# 重新绘制依赖图
shap.dependence_plot(
    heating_idx,
    shap_values,
    X_train_combined,
    feature_names=all_feature_names,
    interaction_index=catalyst_idx,
    alpha=0.7,
    dot_size=16,
    show=False,
    ax=ax2
)

# 设置固定范围防止自动调整留白
ax2.set_xlim(xmin, xmax)
ax2.set_ylim(-110, 70)

# 美化设置
ax2.set_title("", fontsize=14, fontweight='bold')

# 确保所有四个边框都可见并加粗
for spine in ax2.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(2.5)
    
ax2.set_facecolor((1, 1, 1, 0.2))

# 加大坐标轴数字大小和刻度线尺寸
ax2.tick_params(axis='both', width=2.5, length=8, labelsize=18)

# 设置X轴和Y轴标签
ax2.set_xlabel("Heating Rate", fontsize=24, fontweight='bold')
ax2.set_ylabel("SHAP Value", fontsize=24, fontweight='bold')

# 找到并修改颜色条对象
cbar = [child for child in ax2.get_children() if isinstance(child, plt.cm.ScalarMappable)]
if cbar:
    cbar = cbar[0]
    # 获取颜色条对应的坐标轴
    cax = cbar.colorbar.ax
    # 设置颜色条标签
    cax.set_ylabel('Catalyst Mass Fraction', fontsize=22, fontweight='bold', labelpad=20)
    # 设置颜色条刻度大小
    cax.tick_params(labelsize=16, length=5, width=2)
    # 加粗颜色条边框
    for spine in cax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0)

plt.tight_layout()

# 显示所有图形
plt.show()


# In[5]:


import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体为 Arial，并加粗坐标轴标签
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# 创建图像
plt.figure(figsize=(10, 8))

# 假设 y_train, y_test, train_pred, test_pred 已定义
all_values = np.concatenate([y_train, y_test, train_pred, test_pred])
min_val = np.floor(np.min(all_values))  
max_val = np.ceil(np.max(all_values))   

train_point_color = 'skyblue'
test_point_color = 'lightcoral'
train_line_color = '#1E82D6'
test_line_color = '#CC3333'

plt.scatter(y_train, train_pred, c=train_point_color, alpha=0.7,
            edgecolors='w', linewidth=0.5, s=60, label='Train')
plt.scatter(y_test, test_pred, c=test_point_color, alpha=0.7,
            edgecolors='w', linewidth=0.5, s=60, label='Test')

def add_regression_line(x, y, line_color, label_suffix):
    coeffs = np.polyfit(x, y, 1)
    reg_line = np.poly1d(coeffs)
    x_range = np.linspace(np.min(x), np.max(x), 100)
    plt.plot(x_range, reg_line(x_range), color=line_color,
             linestyle='--', lw=2, alpha=0.9,
             label=f'{label_suffix}')
plt.plot([min_val, max_val], [min_val, max_val],
         color='#666666', linewidth=2, linestyle='--',
         alpha=0.9, label='Ideal Prediction')
add_regression_line(y_train, train_pred, train_line_color, 'Train')
add_regression_line(y_test, test_pred, test_line_color, 'Test')



plt.xlabel('Actual $T_{\t{peak}}$(°C)', fontsize=28, labelpad=8)
plt.ylabel('Predicted $T_{\t{peak}}$(°C)', fontsize=28, labelpad=8)
plt.title('', fontsize=14, pad=18, fontweight='semibold')

plt.legend(fontsize=23, 
           framealpha=False, 
           edgecolor='#ffffff',
           loc='upper left',          
           bbox_to_anchor=(0, 1.02),  
           borderpad=1,              
           handletextpad=0.8,        
           borderaxespad=0.8)

ax = plt.gca()
ax.set_axisbelow(True)
ax.set_aspect('equal', adjustable='box')

# 控制刻度样式（字体、大小、加粗、方向）
ax.tick_params(axis='both', which='major', labelsize=28, width=1.5, length=6, direction='out', labelcolor='black')
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontweight('bold')

# 加粗坐标轴边框
for spine in ax.spines.values():
    spine.set_linewidth(2)

plt.tight_layout()
plt.show()


# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

# 设置全局字体为Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 32
plt.rcParams['ytick.labelsize'] = 32
plt.rcParams['axes.labelsize'] = 32
plt.rcParams['legend.fontsize'] = 24
plt.rcParams['ytick.direction'] = 'out'  # Y轴刻度线指向内侧
plt.rcParams['xtick.direction'] = 'out'  # X轴刻度线指向外侧

# 计算残差
train_residuals = y_train - train_pred
test_residuals = y_test - test_pred

# 创建自定义布局的画布
plt.figure(figsize=(11, 8))
# 减少子图间距，使核密度图更贴近散点图
gs = GridSpec(1, 2, width_ratios=[3, 0.8], wspace=0.01)  # 调整宽度比例和间距

# 设置颜色方案
train_color = 'skyblue'
test_color = 'lightcoral'
zero_line_color = 'darkred'  # 鲜艳的红色
grid_color = 'gray'      # 浅灰色网格线

# 1. 残差散点图（x轴从150开始）
ax_scatter = plt.subplot(gs[0])
# 添加水平网格线（首先绘制，确保在数据点下方）
ax_scatter.grid(axis='y', color=grid_color, linestyle='--', linewidth=2, alpha=0.9)
# 绘制数据点
ax_scatter.scatter(train_pred, train_residuals, alpha=0.7, 
                  color=train_color, s=60, label='Train',edgecolors='w', linewidth=0.5)
ax_scatter.scatter(test_pred, test_residuals, alpha=0.7, 
                  color=test_color, s=60, label='Test',edgecolors='w', linewidth=0.5)
# 绘制零线（在网格线上方）
ax_scatter.axhline(y=0, color=zero_line_color, linestyle='--', linewidth=2)

ax_scatter.set_xlabel('Predicted $T_{\t{peak}}$(°C)', fontsize=36, fontweight='bold')
ax_scatter.set_ylabel('Residuals', fontsize=36, fontweight='bold')

# 加粗散点图边框
for spine in ax_scatter.spines.values():
    spine.set_linewidth(2.5)
    spine.set_color('#333333')  # 深灰色边框
    
# 设置X和Y轴刻度线
ax_scatter.tick_params(
    axis='both', 
    width=2, 
    length=8,
    colors='#333333'  # 深灰色刻度
)

# 设置X轴从150开始
ax_scatter.set_xlim(150, max(max(train_pred), max(test_pred)) * 1.05)

# 设置Y轴范围（只显示残差±90内的点）
y_min = -90
y_max = 90
ax_scatter.set_ylim(y_min, y_max)

# 设置残差刻度线
y_locator = ticker.MultipleLocator(30)
ax_scatter.yaxis.set_major_locator(y_locator)

# 禁用科学计数法
ax_scatter.ticklabel_format(useOffset=False, style='plain')
ax_scatter.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

# 添加图例
ax_scatter.legend(frameon=False, edgecolor='#FFFFFF', framealpha=0.9)

# 2. 残差核密度图（贴在散点图右侧）
ax_kde = plt.subplot(gs[1])
sns.kdeplot(y=train_residuals, fill=True, color=train_color, 
           alpha=0.6, linewidth=2.5, ax=ax_kde)
sns.kdeplot(y=test_residuals, fill=True, color=test_color, 
           alpha=0.4, linewidth=2.5, ax=ax_kde)
ax_kde.axhline(y=0, color=zero_line_color, linestyle='--', linewidth=2)

# 设置核密度图的Y轴范围与散点图一致
ax_kde.set_ylim(y_min, y_max)

# 移除核密度图的所有边框和刻度
ax_kde.axis('off')
ax_kde.set_xticks([])
ax_kde.set_yticks([])

# 微调核密度图位置
pos = ax_kde.get_position()
ax_kde.set_position([pos.x0, pos.y0, pos.width, pos.height])

# 调整整体布局
plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.95)
plt.show()


# In[7]:


from scipy import stats
import pandas as pd

def calculate_residual_metrics(residuals, name):
    metrics = {
        'Mean': np.mean(residuals),
        'Median': np.median(residuals),
        'Std Dev': np.std(residuals),
        'Skewness': stats.skew(residuals),
        'Kurtosis': stats.kurtosis(residuals),
        'Min': np.min(residuals),
        '25%': np.percentile(residuals, 25),
        '75%': np.percentile(residuals, 75),
        'Max': np.max(residuals)
    }
    return pd.DataFrame(metrics, index=[name])

# 计算训练集和测试集的残差指标
train_metrics = calculate_residual_metrics(train_residuals, 'Train')
test_metrics = calculate_residual_metrics(test_residuals, 'Test')

# 合并结果并打印
residual_metrics = pd.concat([train_metrics, test_metrics])
print("残差分布统计指标:")
print(residual_metrics.round(2))


# In[8]:


df.dropna(subset=['ΔTemperature_peak'], inplace=True)
y = df['ΔTemperature_peak'].values
excluded_columns = [
    "paper_number", "catalyst_number", "catalysts_component",
    "Temperature_peak(℃)", "ΔTemperature_peak"
]
X = df.drop(excluded_columns, axis=1)  # 修正变量名错误

# 划分训练集和测试集（一九分）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# 定义需要保留的列（不参与降维）
exclude_for_pca = ["Catalysts_Mass_Fraction(wt%)", "heating_rate(℃/min)"]

# 提取保留列
X_train_retained = X_train[exclude_for_pca].values
X_test_retained = X_test[exclude_for_pca].values

# 准备降维数据
X_train_for_pca = X_train.drop(exclude_for_pca, axis=1)
X_test_for_pca = X_test.drop(exclude_for_pca, axis=1)

# 使用PCA降维到60维
pca = PCA(n_components=90, random_state=42)
X_train_pca = pca.fit_transform(X_train_for_pca)
X_test_pca = pca.transform(X_test_for_pca)

# 合并保留列与PCA特征
X_train_combined = np.hstack([X_train_retained, X_train_pca])
X_test_combined = np.hstack([X_test_retained, X_test_pca])

print("训练集合并后维度:", X_train_combined.shape)  # 应为 (样本数, 62)
print("测试集合并后维度:", X_test_combined.shape)


# In[14]:


rf = RandomForestRegressor(n_estimators=1000, random_state=42,max_depth=100,min_samples_split=2,min_samples_leaf=2,criterion='absolute_error',
                           min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, 
                           oob_score=False, n_jobs=-1,verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None)

rf.fit(X_train_combined, y_train)

# 训练集评估（使用X_train_combined）
train_pred = rf.predict(X_train_combined)
print('Training R2 = %.3f' % r2_score(y_train, train_pred))
print('Training RMSE = %.3f' % np.sqrt(mean_squared_error(y_train, train_pred)))
print('Training MAE = %.3f' % mean_absolute_error(y_train, train_pred))

# 测试集评估（使用X_test_combined）
test_pred = rf.predict(X_test_combined)
print('\nTesting R2 = %.3f' % r2_score(y_test, test_pred))
print('Testing RMSE = %.3f' % np.sqrt(mean_squared_error(y_test, test_pred)))
print('Testing MAE = %.3f' % mean_absolute_error(y_test, test_pred))
# 交叉验证（使用X_train_combined）
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(rf, X_train_combined, y_train, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=-1)
rmse_scores = [np.sqrt(abs(s)) for s in scores]
r2_scores = cross_val_score(rf, X_train_combined, y_train, scoring='r2', cv=crossvalidation, n_jobs=-1)
mae_scores = cross_val_score(rf, X_train_combined, y_train, scoring='neg_mean_absolute_error', cv=crossvalidation, n_jobs=-1)

print('\nCross-validation results:')
print('Folds: %i, mean R2: %.3f' % (len(r2_scores), np.mean(r2_scores)))
print('Folds: %i, mean RMSE: %.3f' % (len(rmse_scores), np.mean(rmse_scores)))
print('Folds: %i, mean MAE: %.3f' % (len(mae_scores), np.mean(np.abs(mae_scores))))


# In[10]:


import shap
import matplotlib.pyplot as plt
import numpy as np

modified_names = exclude_for_pca.copy()

# === 特征名修正 ===
modified_names = exclude_for_pca.copy()

# 使用正确的原始特征名进行替换
modified_names = [name.replace('heating_rate(℃/min)', 'Heating Rate') 
                 for name in modified_names]
                 
modified_names = [name.replace('Catalysts_Mass_Fraction(wt%)', 'Cat. Fraction') 
                 for name in modified_names]

# 更新保留的特征名称
retained_feature_names = modified_names

# NMF特征名
pca_feature_names = [f'PCA_{i}' for i in range(X_train_pca.shape[1])]

# 合并所有特征名
all_feature_names = retained_feature_names + pca_feature_names

# 打印验证
print("修正后的特征名称:", all_feature_names[:5])  # 打印前5个验证



# === 绘图样式设置 ===
plt.rcParams.update({
    'figure.facecolor': (0, 0, 0, 0),  # 透明背景
    'axes.facecolor': (0, 0, 0, 0),   # 透明背景
    'axes.linewidth': 2.5,             # 加粗坐标轴线
    'xtick.major.width': 2,            # 加粗x轴刻度线
    'ytick.major.width': 2,            # 加粗y轴刻度线
    'xtick.major.size': 14,             # 加大刻度尺寸
    'ytick.major.size': 14,             # 加大刻度尺寸
    'font.weight': 'bold',             # 加粗字体
    'axes.labelweight': 'bold',        # 加粗轴标签
    'axes.titleweight': 'bold'         # 加粗标题
})

# --- 创建SHAP解释器 ---
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_train_combined)

# ===== 特征重要性图 - 修正顺序问题 =====
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_train_combined, feature_names=all_feature_names, 
                  plot_type="bar", max_display=10, show=False)

# 获取当前坐标轴并设置样式
ax = plt.gca()

# 计算特征重要性值（平均绝对SHAP值）
feature_importance = np.mean(np.abs(shap_values), axis=0)
# 获取前10个最重要的特征索引（按降序排列）
top_indices = np.argsort(feature_importance)[::-1][:10]
 

# 获取所有条形对象（从顶部到底部）
bars = ax.containers[0]

for bar in ax.containers[0]:
    bar.set_color('skyblue')  
    bar.set_alpha(0.6)  # 添加透明度

# 添加重要性数值标签（按从上到下的顺序）
for i in range(len(bars)):
    bar = bars[i]
    # 获取对应特征的重要性值（按降序索引）
    value = feature_importance[top_indices[len(bars)-1-i]]
    
    # 在条形内部左侧添加数值标签
    # 偏移量为条形宽度的1%，确保在条内左侧
    text_x = bar.get_width() * 0.02
    text_y = bar.get_y() + bar.get_height()/2
    
    # 格式化数值（保留两位小数）
    ax.text(text_x, text_y, f'{value:.1f}', 
            fontsize=14, fontweight='bold', color='black',
            verticalalignment='center')

# 将整个横坐标轴（刻度和边框）移动到上方
ax.xaxis.set_ticks_position('top')       # 刻度移动到顶部
ax.xaxis.set_label_position('top')       # 标签移动到顶部
ax.spines['top'].set_position(('axes', 1.0))  # 顶边框保持原位

ax.set_xlabel('Feature Importance', fontsize=20, fontweight='bold', labelpad=10)

# 将横坐标轴线移动到上方位置
ax.spines['bottom'].set_visible(False)   # 隐藏底部轴线
ax.spines['top'].set_visible(True)       # 确保顶部轴线可见
ax.spines['top'].set_linewidth(2.5)       # 加粗顶部轴线
ax.spines['top'].set_zorder(10)          # 确保轴线在最上层

# 调整纵轴位置到左侧
ax.spines['left'].set_position(('axes', 0.0))
ax.yaxis.set_ticks_position('left')
ax.yaxis.set_label_position('left')

# 隐藏右侧和顶部的多余轴线
ax.spines['right'].set_visible(False)

# 设置透明背景和边框线宽
ax.set_facecolor((0, 0, 0, 0))
for spine in ax.spines.values():
    spine.set_linewidth(2.5)

# === 加大刻度尺寸 ===
# 增加刻度线的长度和宽度
ax.tick_params(axis='x', which='major', width=3, size=14)  # x轴刻度（顶部）
ax.tick_params(axis='y', which='major', width=3, size=14)  # y轴刻度（左侧）

# 增加刻度标签的字体大小
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)

plt.title("", fontsize=14, fontweight='bold')
plt.tight_layout()


# ===== 详细特征影响点图 - 只显示前10个 =====
plt.figure(figsize=(12, 8), dpi=100)  # 增加DPI以获得更清晰的渲染
# 绘制SHAP点图，但不显示（show=False）
shap.summary_plot(
    shap_values, 
    X_train_combined, 
    feature_names=all_feature_names, 
    max_display=10, 
    show=False
)

# 获取当前坐标轴
ax2 = plt.gca()

# 设置横坐标轴名称
ax2.set_xlabel('SHAP Value', fontsize=20, fontweight='bold', labelpad=10)

# 双重保障设置标签
if ax2.get_xlabel() == '':
    ax2.set_xlabel('SHAP Value', fontsize=20, fontweight='bold', labelpad=10)

# 加大刻度尺寸
ax2.tick_params(axis='x', which='major', width=3, size=14)
ax2.tick_params(axis='y', which='major', width=3, size=14)
ax2.tick_params(axis='x', labelsize=16)
ax2.tick_params(axis='y', labelsize=16)

# === 可靠方法加大颜色条字体 ===
# 1. 获取所有axes
all_axes = plt.gcf().get_axes()

# 2. 查找颜色条对象
cbar = None
for ax in all_axes:
    # 尝试获取颜色条对象
    if hasattr(ax, 'collections') and len(ax.collections) > 0:
        # 查找散点图集合
        for collection in ax.collections:
            if hasattr(collection, 'colorbar'):
                cbar = collection.colorbar
                break
    if cbar is not None:
        break

# 3. 如果找到颜色条，直接修改其属性
if cbar is not None:
    # 方法1：直接修改颜色条刻度标签
    cbar.ax.tick_params(labelsize=16)
    
    # 方法2：重新设置颜色条标签
    cbar.set_label(cbar.ax.get_ylabel(), size=16, weight='bold')
    
    # 方法3：强制更新颜色条
    cbar.update_normal(cbar.mappable)

# 4. 如果上述方法失败，使用备用方法
if cbar is None:
    # 查找所有颜色条axes
    for ax in all_axes:
        if 'colorbar' in ax.get_label().lower():
            # 加大颜色条刻度标签字体
            ax.tick_params(labelsize=16)
            
            # 加大颜色条标题字体
            cbar_label = ax.get_ylabel()
            ax.set_ylabel(cbar_label, fontsize=16, fontweight='bold')
            
            # 强制重绘
            ax.figure.canvas.draw_idle()
            break

# 确保标题为空
plt.title("", fontsize=14, fontweight='bold')

# 手动调整布局
plt.tight_layout(rect=[0, 0, 1, 0.97])  # 为顶部标签留出空间

# 显式刷新图形 - 使用更强大的方法
plt.gcf().canvas.draw()
plt.gcf().canvas.flush_events()

# ===== 单样本解释 ===== 
plt.figure(figsize=(12, 4))
shap.force_plot(
    explainer.expected_value,
    shap_values[0, :],
    X_train_combined[0, :],
    feature_names=all_feature_names,
    matplotlib=True,
    text_rotation=15,
    show=False
)
plt.title("Individual Prediction Explanation", fontsize=14, fontweight='bold')
ax = plt.gca()
ax.set_facecolor((0, 0, 0, 0))
# 确保所有四个边框都可见并加粗
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(2.5)
plt.tight_layout()

# ===== 交互作用分析 - 双视角展示 =====
catalyst_idx = all_feature_names.index('Cat. Fraction')
heating_idx = all_feature_names.index('Heating Rate')

# 视角1：以催化剂负载量为主，加热速率为交互项
fig1, ax1 = plt.subplots(figsize=(10, 6))

# 首先绘制依赖图以获取数据范围
shap.dependence_plot(
    catalyst_idx,
    shap_values,
    X_train_combined,
    feature_names=all_feature_names,
    interaction_index=heating_idx,
    alpha=0.7,
    dot_size=16,
    show=False,
    ax=ax1
)

# 获取数据范围（包括padding）
xmin_data, xmax_data = ax1.get_xlim()
ymin_data, ymax_data = ax1.get_ylim()

# 计算扩展范围（消除边距）
padding_factor = 0.02  # 无padding
xmin = xmin_data - padding_factor * (xmax_data - xmin_data)
xmax = xmax_data + padding_factor * (xmax_data - xmin_data)
ymin = ymin_data - padding_factor * (ymax_data - ymin_data)
ymax = ymax_data + padding_factor * (ymax_data - ymin_data)

# 清除当前图形
ax1.clear()

# 设置背景分区 - 使用扩展的范围确保完全覆盖
# 小于5的区域 - 淡灰色
ax1.axvspan(xmin=xmin, xmax=4.8, color='lightgray', alpha=0.45, zorder=0)
# 5-10的区域 - 淡红色
ax1.axvspan(xmin=5, xmax=9.8, color='lightblue', alpha=0.35, zorder=0)
# 大于10的区域 - 淡蓝色
ax1.axvspan(xmin=10, xmax=xmax, color='lightcoral', alpha=0.3, zorder=0)

# 添加SHAP=0的黑色虚线，加粗
ax1.axhline(y=0, color='black', linestyle='--', linewidth=2.5, zorder=1)

# 重新绘制依赖图
shap.dependence_plot(
    catalyst_idx,
    shap_values,
    X_train_combined,
    feature_names=all_feature_names,
    interaction_index=heating_idx,
    alpha=0.7,
    dot_size=16,
    show=False,
    ax=ax1
)

# 设置固定范围防止自动调整留白
ax1.set_xlim(xmin, xmax)
ax1.set_ylim(ymin, ymax)

# 美化设置
ax1.set_title("", fontsize=14, fontweight='bold')  # 标题为空白

# 确保所有四个边框都可见并加粗
for spine in ax1.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(2.5)
    
ax1.set_facecolor((1, 1, 1, 0.2))

# 加大坐标轴数字大小和刻度线尺寸
ax1.tick_params(axis='both', width=2.5, length=8, labelsize=18)

# 设置X轴和Y轴标签
ax1.set_xlabel("Catalyst Mass Fraction", fontsize=24, fontweight='bold')
ax1.set_ylabel("SHAP Value", fontsize=24, fontweight='bold')

# 找到并修改颜色条对象
cbar = [child for child in ax1.get_children() if isinstance(child, plt.cm.ScalarMappable)]
if cbar:
    cbar = cbar[0]
    # 获取颜色条对应的坐标轴
    cax = cbar.colorbar.ax
    # 设置颜色条标签
    cax.set_ylabel('Heating Rate', fontsize=22, fontweight='bold', labelpad=20)
    # 设置颜色条刻度大小
    cax.tick_params(labelsize=16, length=5, width=2)
    # 加粗颜色条边框
    for spine in cax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0)
        
plt.tight_layout()


# 视角2：以加热速率为主，催化剂负载量为交互项
fig2, ax2 = plt.subplots(figsize=(10, 6))

# 首先绘制依赖图获取范围
shap.dependence_plot(
    heating_idx,
    shap_values,
    X_train_combined,
    feature_names=all_feature_names,
    interaction_index=catalyst_idx,
    alpha=0.7,
    dot_size=16,
    show=False,
    ax=ax2
)

# 获取数据范围（包括padding）
xmin_data, xmax_data = ax2.get_xlim()
ymin_data, ymax_data = ax2.get_ylim()

# 计算扩展范围（消除边距）
padding_factor = 0.2
xmin = xmin_data - padding_factor * (xmax_data - xmin_data)
xmax = xmax_data + padding_factor * (xmax_data - xmin_data)
ymin = ymin_data - padding_factor * (ymax_data - ymin_data)
ymax = ymax_data + padding_factor * (ymax_data - ymin_data)

# 清除当前图形
ax2.clear()

# 设置背景分区 - 以10为分界
# 加热速率小于10的区域 - 淡蓝色
ax2.axvspan(xmin=xmin, xmax=10, color='lightblue', alpha=0.35, zorder=0)
# 加热速率大于等于10的区域 - 淡红色
ax2.axvspan(xmin=10, xmax=xmax, color='lightcoral', alpha=0.3, zorder=0)

# 添加SHAP=0的黑色虚线，加粗
ax2.axhline(y=0, color='black', linestyle='--', linewidth=2.5, zorder=1)

# 重新绘制依赖图
shap.dependence_plot(
    heating_idx,
    shap_values,
    X_train_combined,
    feature_names=all_feature_names,
    interaction_index=catalyst_idx,
    alpha=0.7,
    dot_size=16,
    show=False,
    ax=ax2
)

# 设置固定范围防止自动调整留白
ax2.set_xlim(xmin, xmax)
ax2.set_ylim(-30, 30)

# 美化设置
ax2.set_title("", fontsize=14, fontweight='bold')

# 确保所有四个边框都可见并加粗
for spine in ax2.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(2.5)
    
ax2.set_facecolor((1, 1, 1, 0.2))

# 加大坐标轴数字大小和刻度线尺寸
ax2.tick_params(axis='both', width=2.5, length=8, labelsize=18)

# 设置X轴和Y轴标签
ax2.set_xlabel("Heating Rate", fontsize=24, fontweight='bold')
ax2.set_ylabel("SHAP Value", fontsize=24, fontweight='bold')

# 找到并修改颜色条对象
cbar = [child for child in ax2.get_children() if isinstance(child, plt.cm.ScalarMappable)]
if cbar:
    cbar = cbar[0]
    # 获取颜色条对应的坐标轴
    cax = cbar.colorbar.ax
    # 设置颜色条标签
    cax.set_ylabel('Catalyst Mass Fraction', fontsize=22, fontweight='bold', labelpad=20)
    # 设置颜色条刻度大小
    cax.tick_params(labelsize=16, length=5, width=2)
    # 加粗颜色条边框
    for spine in cax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0)

plt.tight_layout()

# 显示所有图形
plt.show()


# In[11]:


import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体为 Arial，并加粗坐标轴标签
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# 创建图像
plt.figure(figsize=(10, 8))

# 假设 y_train, y_test, train_pred, test_pred 已定义
all_values = np.concatenate([y_train, y_test, train_pred, test_pred])
min_val = np.floor(np.min(all_values))  
max_val = np.ceil(np.max(all_values))   

train_point_color = 'skyblue'
test_point_color = 'lightcoral'
train_line_color = '#1E82D6'
test_line_color = '#CC3333'

plt.scatter(y_train, train_pred, c=train_point_color, alpha=0.7,
            edgecolors='w', linewidth=0.5, s=60, label='Train')
plt.scatter(y_test, test_pred, c=test_point_color, alpha=0.7,
            edgecolors='w', linewidth=0.5, s=60, label='Test')

def add_regression_line(x, y, line_color, label_suffix):
    coeffs = np.polyfit(x, y, 1)
    reg_line = np.poly1d(coeffs)
    x_range = np.linspace(np.min(x), np.max(x), 100)
    plt.plot(x_range, reg_line(x_range), color=line_color,
             linestyle='--', lw=2, alpha=0.9,
             label=f'{label_suffix}')
plt.plot([min_val, max_val], [min_val, max_val],
         color='#666666', linewidth=2, linestyle='--',
         alpha=0.9, label='Ideal Prediction')
add_regression_line(y_train, train_pred, train_line_color, 'Train')
add_regression_line(y_test, test_pred, test_line_color, 'Test')



plt.xlabel('Actual Δ$T_{\t{peak}}$(°C)', fontsize=28, labelpad=8)
plt.ylabel('Predicted Δ$T_{\t{peak}}$(°C)', fontsize=28, labelpad=8)
plt.title('', fontsize=14, pad=18, fontweight='semibold')

plt.legend(fontsize=23, 
           framealpha=False, 
           edgecolor='#ffffff',
           loc='upper left',          
           bbox_to_anchor=(0, 1.02),  
           borderpad=1,              
           handletextpad=0.8,        
           borderaxespad=0.8)

ax = plt.gca()
ax.set_axisbelow(True)
ax.set_aspect('equal', adjustable='box')

# 控制刻度样式（字体、大小、加粗、方向）
ax.tick_params(axis='both', which='major', labelsize=28, width=1.5, length=6, direction='out', labelcolor='black')
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontweight('bold')

# 加粗坐标轴边框
for spine in ax.spines.values():
    spine.set_linewidth(2)

plt.tight_layout()
plt.show()


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

# 设置全局字体为Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 32
plt.rcParams['ytick.labelsize'] = 32
plt.rcParams['axes.labelsize'] = 32
plt.rcParams['legend.fontsize'] = 24
plt.rcParams['ytick.direction'] = 'out'  # Y轴刻度线指向内侧
plt.rcParams['xtick.direction'] = 'out'  # X轴刻度线指向外侧

# 计算残差
train_residuals = y_train - train_pred
test_residuals = y_test - test_pred

# 创建自定义布局的画布
plt.figure(figsize=(11, 8))
# 减少子图间距，使核密度图更贴近散点图
gs = GridSpec(1, 2, width_ratios=[3, 0.8], wspace=0.01)  # 调整宽度比例和间距

# 设置颜色方案
train_color = 'skyblue'
test_color = 'lightcoral'
zero_line_color = 'darkred'  # 鲜艳的红色
grid_color = 'gray'      # 浅灰色网格线

# 1. 残差散点图（x轴从150开始）
ax_scatter = plt.subplot(gs[0])
# 添加水平网格线（首先绘制，确保在数据点下方）
ax_scatter.grid(axis='y', color=grid_color, linestyle='--', linewidth=2, alpha=0.9)
# 绘制数据点
ax_scatter.scatter(train_pred, train_residuals, alpha=0.7, 
                  color=train_color, s=60, label='Train',edgecolors='w', linewidth=0.5)
ax_scatter.scatter(test_pred, test_residuals, alpha=0.7, 
                  color=test_color, s=60, label='Test',edgecolors='w', linewidth=0.5)
# 绘制零线（在网格线上方）
ax_scatter.axhline(y=0, color=zero_line_color, linestyle='--', linewidth=2)

ax_scatter.set_xlabel('Predicted Δ$T_{\t{peak}}$(°C)', fontsize=36, fontweight='bold')
ax_scatter.set_ylabel('Residuals', fontsize=36, fontweight='bold')

# 加粗散点图边框
for spine in ax_scatter.spines.values():
    spine.set_linewidth(2.5)
    spine.set_color('#333333')  # 深灰色边框
    
# 设置X和Y轴刻度线
ax_scatter.tick_params(
    axis='both', 
    width=2, 
    length=8,
    colors='#333333'  # 深灰色刻度
)

# 设置X轴从150开始
ax_scatter.set_xlim(-20, max(max(train_pred), max(test_pred)) * 1.05)

# 设置Y轴范围（只显示残差±90内的点）
y_min = -45
y_max = 45
ax_scatter.set_ylim(y_min, y_max)

# 设置残差刻度线
y_locator = ticker.MultipleLocator(15)
ax_scatter.yaxis.set_major_locator(y_locator)

# 禁用科学计数法
ax_scatter.ticklabel_format(useOffset=False, style='plain')
ax_scatter.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

# 添加图例
ax_scatter.legend(frameon=False, edgecolor='#FFFFFF', framealpha=0.9)

# 2. 残差核密度图（贴在散点图右侧）
ax_kde = plt.subplot(gs[1])
sns.kdeplot(y=train_residuals, fill=True, color=train_color, 
           alpha=0.6, linewidth=2.5, ax=ax_kde)
sns.kdeplot(y=test_residuals, fill=True, color=test_color, 
           alpha=0.4, linewidth=2.5, ax=ax_kde)
ax_kde.axhline(y=0, color=zero_line_color, linestyle='--', linewidth=2)

# 设置核密度图的Y轴范围与散点图一致
ax_kde.set_ylim(y_min, y_max)

# 移除核密度图的所有边框和刻度
ax_kde.axis('off')
ax_kde.set_xticks([])
ax_kde.set_yticks([])

# 微调核密度图位置
pos = ax_kde.get_position()
ax_kde.set_position([pos.x0, pos.y0, pos.width, pos.height])

# 调整整体布局
plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.95)
plt.show()


# In[13]:


from scipy import stats
import pandas as pd

def calculate_residual_metrics(residuals, name):
    metrics = {
        'Mean': np.mean(residuals),
        'Median': np.median(residuals),
        'Std Dev': np.std(residuals),
        'Skewness': stats.skew(residuals),
        'Kurtosis': stats.kurtosis(residuals),
        'Min': np.min(residuals),
        '25%': np.percentile(residuals, 25),
        '75%': np.percentile(residuals, 75),
        'Max': np.max(residuals)
    }
    return pd.DataFrame(metrics, index=[name])

# 计算训练集和测试集的残差指标
train_metrics = calculate_residual_metrics(train_residuals, 'Train')
test_metrics = calculate_residual_metrics(test_residuals, 'Test')

# 合并结果并打印
residual_metrics = pd.concat([train_metrics, test_metrics])
print("残差分布统计指标:")
print(residual_metrics.round(2))

