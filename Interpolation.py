import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import cpu_count

# 配置参数
MAX_ANGLE = 180.0
STEP = 0.02
MIN_PEAK_HEIGHT = 0.1
TOP_PEAKS = 20
N_CORES = cpu_count()  # 自动获取可用核心数

# 输入输出路径
input_dir = ""
output_dir = ""
plot_dir = ""

os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

def enhanced_interpolation(theta_orig, intensity_orig):
    """优化后的插值函数（包含归一化处理）"""
    # 数据清洗
    valid_mask = (~np.isnan(intensity_orig)) & (intensity_orig > 1e-6)
    theta_clean = theta_orig[valid_mask]
    intensity_clean = intensity_orig[valid_mask]
    
    if len(theta_clean) < 2:
        return np.array([]), np.array([])
    
    # 强度归一化 (缩放到0-1范围)
    if np.max(intensity_clean) > 0:
        intensity_clean = intensity_clean / np.max(intensity_clean)
    
    # 边界扩展配置
    EXTEND_STEPS = 5  # 每个方向扩展点数
    
    # 自动生成0.02°整数倍的网格 ================ 修改部分 ================
    # 计算网格起点（向下舍入到最近0.02°倍数）
    min_theta = max(0, np.min(theta_clean))
    start_point = np.floor(min_theta * 50) / 50.0  # 50=1/0.02
    
    # 计算网格终点（向上舍入到最近0.02°倍数）
    max_theta = min(MAX_ANGLE, np.max(theta_clean))
    end_point = np.ceil(max_theta * 50) / 50.0
    
    # 扩展起点和终点（按0.02°步长扩展）
    extended_start = max(0, start_point - EXTEND_STEPS * STEP)
    extended_end = min(MAX_ANGLE, end_point + EXTEND_STEPS * STEP)
    
    # 生成精确对齐的网格（确保所有点都是0.02°的整数倍）
    num_points = int(round((extended_end - extended_start) / STEP)) + 1
    theta_new = np.linspace(extended_start, extended_end, num_points)
    theta_new = np.round(theta_new, 2)  # 确保精确两位小数
    # =====================================================
    
    # 局部扩展插值
    interp_func = interp1d(
        theta_clean,
        intensity_clean,
        kind='linear',
        bounds_error=False,
        fill_value=0.0
    )
    
    intensity_new = interp_func(theta_new)
    intensity_new = np.clip(intensity_new, 0, 1.0)  # 确保值在0-1范围内
    
    return theta_new, intensity_new

def normalize_intensity(intensity):
    """归一化强度数据到0-1范围"""
    # 数据清洗
    valid_mask = (~np.isnan(intensity)) & (intensity > 1e-6)
    intensity_clean = intensity[valid_mask]
    
    if len(intensity_clean) == 0:
        return np.zeros_like(intensity)
    
    # 强度归一化
    max_val = np.max(intensity_clean)
    if max_val > 0:
        return intensity / max_val
    return np.zeros_like(intensity)

def process_file(input_path, output_path, plot_path=None):
    try:
        df = pd.read_excel(input_path)
        formula = df['chemical_formula'].iloc[0]
        
        # 归一化原始数据用于绘图
        orig_intensity_normalized = normalize_intensity(df['intensity'].values.astype(float))
        
        # 生成插值数据
        theta_new, intensity_new = enhanced_interpolation(
            df['theta'].values.astype(float),
            df['intensity'].values.astype(float)
        )
        
        # 跳过无效数据
        if len(theta_new) == 0:
            return False
        
        # 构建输出DataFrame
        output_df = pd.DataFrame({
            'theta': theta_new,
            'intensity': np.round(intensity_new, 6),
            'chemical_formula': [formula] * len(theta_new)
        })
        
        output_df.to_csv(output_path, index=False)
        
        # 生成对比图（前500个文件）
        if plot_path and os.path.basename(input_path) < 'file_500.xlsx':
            plt.figure(figsize=(14, 7))
            
            # 原始数据：使用归一化后的强度
            plt.plot(df['theta'], orig_intensity_normalized, 
                    color='#1f77b4',
                    linestyle='-',
                    linewidth=1.5,
                    marker='o',
                    markersize=4,
                    markeredgecolor='k',
                    alpha=0.8,
                    label='Original Data (Normalized)')
            
            # 插值数据
            plt.scatter(theta_new, intensity_new,
                      color='#ff7f0e',
                      s=10,
                      alpha=0.6,
                      edgecolors='none',
                      label='Interpolated Points',
                      zorder=3)
            
            plt.title(f"XRD Pattern - {formula}", fontsize=14, pad=15)
            plt.xlabel("2θ (degrees)", fontsize=12)
            plt.ylabel("Normalized Intensity", fontsize=12)
            plt.legend(loc='upper right', fontsize=10)
            plt.grid(True, linestyle=':', color='gray', alpha=0.6)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=200, bbox_inches='tight')
            plt.close()
            
        return True
    except Exception as e:
        print(f"Error processing {os.path.basename(input_path)}: {str(e)}")
        return False

def main():
    # 准备任务列表
    file_list = sorted([f for f in os.listdir(input_dir) if f.endswith('.xlsx')])
    tasks = []
    
    for idx, filename in enumerate(file_list):
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename.replace('.xlsx', '.csv'))
        plot_path = os.path.join(plot_dir, f"{filename[:-5]}.png") if idx < 500 else None
        tasks.append((in_path, out_path, plot_path))
    
    # 并行处理
    print(f"Starting processing {len(tasks)} files with {N_CORES} cores...")
    results = Parallel(n_jobs=N_CORES, verbose=10)(
        delayed(process_file)(*task) for task in tasks
    )
    
    # 输出统计
    success = sum(results)
    print(f"\nProcessing completed: {success} successes, {len(tasks)-success} failures")
    print(f"Results saved to: {output_dir}")
    print(f"Plots saved to: {plot_dir}")

if __name__ == "__main__":
    main()