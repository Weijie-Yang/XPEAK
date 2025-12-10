import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import find_peaks
import math
import os
from sklearn.manifold import TSNE

def process_xrd_csv(file_path):
    # 基于您提供的代码修改，读取本地文件而不是 stream
    print(f"[BACKEND] Processing file: {file_path}")
    try:
        df = pd.read_csv(file_path, header=None)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        raise ValueError(f"无法解析CSV文件。请检查文件内容和格式。错误: {e}")

    # 确保数据有两列，且没有空值
    if len(df.columns) < 2:
        raise ValueError("CSV文件必须包含至少两列数据（角度和强度）")
    
    df = df.iloc[:, :2]
    df.columns = ['angle', 'intensity']
    
    if df.isnull().values.any():
        print("警告：数据中包含空值，使用0填充...")
        df = df.fillna(0)
    
    try:
        df['angle'] = df['angle'].astype(float)
        df['intensity'] = df['intensity'].astype(float)
    except (ValueError, TypeError) as e:
        print(f"数据类型转换错误: {e}")
        raise ValueError(f"角度或强度列包含非数字值。请检查数据格式。")
        
    if np.isinf(df['angle']).any() or np.isinf(df['intensity']).any():
        print("警告：数据中包含无限大值，移除这些行...")
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # 第一次归一化
    scaler = MinMaxScaler()
    intensity_values = df['intensity'].values.reshape(-1, 1)
    
    if np.max(intensity_values) == np.min(intensity_values) == 0:
        print("警告：所有强度值均为0，无法归一化，跳过此步骤")
        df['intensity'] = 0
    else:
        df['intensity'] = scaler.fit_transform(intensity_values).flatten()

    # 精确到0.02整数倍的插值
    print("[BACKEND] Step 2: Performing 0.02-aligned peak-preserving interpolation...")
    df_sorted = df.groupby('angle', as_index=False).agg({'intensity': 'mean'})
    x = df_sorted['angle'].values
    y = df_sorted['intensity'].values

    if len(y) == 0:
        raise ValueError("处理后数据为空，无法继续处理")
        
    peaks, _ = find_peaks(y)
    num_peaks_to_get = min(20, len(peaks))
    
    if num_peaks_to_get > 0:
        top_20_peaks_indices = peaks[np.argsort(y[peaks])[-num_peaks_to_get:][::-1]]
        top_20_peaks = y[top_20_peaks_indices]
        top_20_peaks_positions = x[top_20_peaks_indices]
    else:
        top_20_peaks = np.array([])
        top_20_peaks_positions = np.array([])
        print("警告：未检测到任何峰值")

    print("Top 20 Peaks:")
    for pos, value in zip(top_20_peaks_positions, top_20_peaks):
        print(f"Position: {pos:.2f}, Intensity: {value:.4f}")

    min_angle = math.floor(df['angle'].min() * 50) / 50
    max_angle = math.ceil(df['angle'].max() * 50) / 50
    x_new = np.round(np.arange(min_angle, max_angle + 0.02, 0.02), 2)
    
    print(f"插值网格: 起点={x_new[0]:.2f}, 终点={x_new[-1]:.2f}, 步长=0.02, 点数={len(x_new)}")

    if len(x) >= 2:
        x_extended = np.concatenate([[min_angle - 0.02], x, [max_angle + 0.02]])
        y_extended = np.concatenate([[y[0]], y, [y[-1]]])
    else:
        x_extended = np.concatenate([x, [x[0] + 0.02]])
        y_extended = np.concatenate([y, [y[0]]])

    f_linear = interp1d(x_extended, y_extended, kind='linear', fill_value="extrapolate")
    y_new = f_linear(x_new)

    peak_dict = {}
    if len(top_20_peaks_positions) > 0:
        for pos, value in zip(top_20_peaks_positions, top_20_peaks):
            idx = np.argmin(np.abs(x_new - pos))
            if idx not in peak_dict or value > peak_dict[idx]:
                peak_dict[idx] = value

        for idx, value in peak_dict.items():
            y_new[idx] = value

    df_interp = pd.DataFrame({'angle': x_new, 'intensity': y_new})

    if not (np.max(df_interp['intensity']) == 0 and np.min(df_interp['intensity']) == 0):
        intensity_interp = df_interp['intensity'].values.reshape(-1, 1)
        df_interp['intensity'] = scaler.fit_transform(intensity_interp).flatten()

    # 构建0.01对齐的特征矩阵 (10.00-90.00)
    print("[BACKEND] Step 3: Building 0.01-aligned feature matrix in [10.00, 90.00] range...")
    start_angle = 10.00
    end_angle = 90.00
    step = 0.01
    num_points = int((end_angle - start_angle) / step) + 1
    
    standard_angles_float = np.linspace(start_angle, end_angle, num_points)
    standard_angles_float = np.round(standard_angles_float / step) * step
    standard_angles_float = np.round(standard_angles_float, 2)
    standard_angles_str = [f"{x:.2f}" for x in standard_angles_float]
    
    print(f"标准角度范围: {start_angle:.2f}到{end_angle:.2f}, 步长={step}, 点数={len(standard_angles_float)}")
    print(f"首尾角度值: {standard_angles_float[0]}, {standard_angles_float[-1]}")

    df_interp['angle_rounded'] = np.round(df_interp['angle'].astype(float) / step) * step
    df_interp['angle_rounded'] = df_interp['angle_rounded'].map("{:.2f}".format)
    
    series_interp = df_interp.set_index('angle_rounded')['intensity']
    final_series = pd.Series(index=standard_angles_str, dtype='float64')
    final_series = final_series.fillna(0)
    final_series.update(series_interp)
    final_series = final_series.fillna(0)
    
    if final_series.isnull().any():
        print("警告：特征矩阵中存在None值，将其替换为0")
        final_series = final_series.fillna(0)
        
    feature_matrix_xrd = pd.DataFrame([final_series.values], columns=standard_angles_str)
    feature_matrix_xrd = feature_matrix_xrd[standard_angles_str]
    
    return feature_matrix_xrd

def extend_features_to_match(features, target_dim, method='constant'):
    """
    将特征扩展到目标维度
    
    参数:
    features: 原始特征数组 (n_samples, n_features)
    target_dim: 目标特征维度
    method: 扩展方法 ('constant': 用0填充, 'edge': 用边界值填充)
    """
    current_dim = features.shape[1]
    
    if current_dim == target_dim:
        return features
    
    print(f"扩展特征维度: {current_dim} -> {target_dim}, 方法: {method}")
    
    if current_dim > target_dim:
        # 如果当前维度大于目标维度，截断
        print(f"警告: 当前维度({current_dim})大于目标维度({target_dim})，将进行截断")
        return features[:, :target_dim]
    
    # 当前维度小于目标维度，需要扩展
    extended_features = np.zeros((features.shape[0], target_dim))
    extended_features[:, :current_dim] = features
    
    if method == 'edge' and current_dim > 0:
        # 使用边界值填充
        edge_value = features[:, -1]  # 使用最后一个值作为边界值
        for i in range(current_dim, target_dim):
            extended_features[:, i] = edge_value
        print("使用边界值扩展")
    else:
        # 使用0填充（默认）
        print("使用0值扩展")
    
    return extended_features

# Main processing logic
paper_folder = ''
train_file = ''

# Process all CSV files
csv_files = [f for f in os.listdir(paper_folder) if f.endswith('.csv')]
paper_data = []

print(f"在paper文件夹中找到 {len(csv_files)} 个CSV文件")

# 处理所有CSV文件
for file in csv_files:
    file_path = os.path.join(paper_folder, file)
    print(f"处理文件: {file}")
    matrix = process_xrd_csv(file_path)
    paper_data.append(matrix.iloc[0].values)
    print(f"文件 {file} 处理完成，特征维度: {matrix.shape[1]}")

paper_features = np.array(paper_data)  # shape: (num_paper_samples, num_features)
print(f"实验数据特征维度: {paper_features.shape}")

# 读取训练数据
train_df = pd.read_excel(train_file)
columns = train_df.columns
print(f"Excel文件列名: {list(columns)}")

# 自动检测特征列的开始位置
start_col_index = None
for i, col in enumerate(columns):
    try:
        # 尝试将列名转换为浮点数，如果是角度值（10.00, 10.01等）
        angle_val = float(str(col).strip())
        if 10.0 <= angle_val <= 90.0:  # 角度通常在10-90度范围内
            start_col_index = i
            print(f"在索引 {i} 找到角度列: '{col}' = {angle_val}")
            break
    except ValueError:
        continue

if start_col_index is None:
    # 如果没有找到角度列，使用默认的第8列（索引7）
    start_col_index = 7
    print(f"未找到角度列，使用默认起始列索引: {start_col_index}")

feature_cols = columns[start_col_index:]
print(f"使用从索引 {start_col_index} 开始的特征列: {list(feature_cols[:5])}...")  # 显示前5个特征列

train_features = train_df[feature_cols].values  # shape: (num_train_samples, num_features)
print(f"训练数据特征维度: {train_features.shape}")

# 检查特征维度是否匹配
print(f"实验数据特征维度: {paper_features.shape[1]}")
print(f"训练数据特征维度: {train_features.shape[1]}")

# 确定目标维度（使用训练数据的维度）
target_dim = train_features.shape[1]
print(f"目标特征维度: {target_dim}")

# 如果维度不匹配，进行扩展
if paper_features.shape[1] != target_dim:
    print("特征维度不匹配，进行扩展...")
    paper_features = extend_features_to_match(paper_features, target_dim, method='edge')

# 再次检查维度
if paper_features.shape[1] != train_features.shape[1]:
    raise ValueError(f"特征维度扩展后仍不匹配: 实验数据={paper_features.shape[1]}, 训练数据={train_features.shape[1]}")

print(f"扩展后实验数据特征维度: {paper_features.shape}")
print(f"训练数据特征维度: {train_features.shape}")

# 合并数据
all_features = np.vstack((train_features, paper_features))
labels = np.array([0] * train_features.shape[0] + [1] * paper_features.shape[0])  # 0: train (red), 1: paper (blue)

print(f"合并后数据维度: {all_features.shape}")
print(f"标签分布: 总数={len(labels)}, 训练数据={np.sum(labels == 0)}, 实验数据={np.sum(labels == 1)}")

# 运行t-SNE
print("运行t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, learning_rate=250, n_iter=1000, random_state=42, init='pca')
tsne_results = tsne.fit_transform(all_features)

print("t-SNE完成，创建可视化...")

# 使用所有数据点进行可视化
tsne_plot = tsne_results
labels_plot = labels

# 可视化
plt.figure(figsize=(10, 9))
plt.rcParams['font.family'] = 'Arial'  # 设置字体为Arial

# 绘制训练数据（红色圆圈，浅灰色边框）
train_mask = labels_plot == 0
plt.scatter(tsne_plot[train_mask, 0], tsne_plot[train_mask, 1], 
            c='lightcoral', marker='o', label='Train', s=70, alpha=0.7,
            edgecolors='lightgray', linewidth=0.5)

# 绘制所有实验数据（蓝色星形，深灰色边框）- 向左下移动
paper_mask = labels_plot == 1
 
plt.scatter(c='#1E90FF', marker='*', label='Validation', s=300, 
            edgecolors='lightgray', linewidth=0.5)

# 自定义图形
plt.xlabel('t-SNE 1', fontsize=24, fontweight='bold')
plt.ylabel('t-SNE 2', fontsize=24, fontweight='bold')
plt.legend(fontsize=30, prop={'size': 15, 'weight': 'bold'}, frameon=False)
plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)  # 移除刻度
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(2.5)  # 加粗图形边框

# 保存图形
plt.savefig('', dpi=640, bbox_inches='tight')
plt.close()

print("包含所有数据点的t-SNE图保存成功！")