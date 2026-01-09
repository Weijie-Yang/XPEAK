import os
import numpy as np
import pandas as pd
from scipy import special, signal
from concurrent.futures import ProcessPoolExecutor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from dataclasses import dataclass
import time
import json
from scipy.stats import skewnorm

# ----------------------
# 增强型配置类
# ----------------------
@dataclass
class EnhancedSimulationConfig:
    # 路径配置
    input_folder: str = ""
    output_folder: str = ""
    plot_folder: str = ""
    
    # 处理参数
    num_workers: int = 120
    top_peaks: int = 20
    demo_num: int = 10
    
    # 物理参数
    wavelength: float = 0.154056  # Cu Kα波长 (nm)
    U: float = 0.05               # Caglioti参数
    V: float = -0.01
    W: float = 0.01
    atomic_displacement: float = 0.02  # 原子位移参数 (Å²)
    
    # 参数范围
    param_ranges: dict = None
    
    def __post_init__(self):
        self.param_ranges = {
            'global_shift': (-0.2, 0.2),
            'crystallite_size': (5, 200),      # 晶粒尺寸 (nm)
            'strain': (0.002, 0.01),         # 应变值
            'split_angle': (0.2, 0.4),
            'preferred_factor': (1.2, 3.0),   # 优化的取向因子范围
            'noise_level': (0.005, 0.03),     # 噪声水平范围
            'sub_size_ratio': (0.2, 0.4)      # 新增子区间比例范围
        }

# ----------------------
# 增强型模拟器类（修复序列化问题）
# ----------------------
class EnhancedXRDSimulator:
    def __init__(self, config: EnhancedSimulationConfig):
        self.config = config
        # 改进的随机种子初始化
        # 更复杂的种子生成：进程ID + 高精度时间 + 对象哈希
        seed = (
            os.getpid() * 0x10001 + 
            int(time.perf_counter() * 1e9) % 0x100000 + 
            hash(self) % 1000
        )
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self._init_folders()
        self.distributions = self._create_distributions()

    def _init_folders(self):
        """创建输出目录"""
        os.makedirs(self.config.output_folder, exist_ok=True)
        os.makedirs(self.config.plot_folder, exist_ok=True)

    def _create_distributions(self):
        """可序列化的分布生成器"""
        return {
            'strain': self._strain_distribution,
            'split_prob': self._split_prob_distribution,
            'noise': self._noise_distribution
        }

    # ----------------------
    # 子区间生成优化
    # ----------------------
    def _generate_sub_range(self, material_idx):
        # 保存当前随机状态
        original_state = self.rng.bit_generator.state
        # 根据 material_idx 生成唯一种子
        seed = int(material_idx) + hash(self) % 1000
        self.rng = np.random.Generator(np.random.PCG64(seed))
        """为每个材料生成独立的差异化子区间（进程安全版）"""
        # 区域配置（显式使用float类型）
        regions = [
            ('low', 5.0, 100.0, 0.4),     # 扩大覆盖范围
            ('mid', 30.0, 150.0, 0.3),
            ('high', 80.0, 200.0, 0.3)    # 增加重叠区域
        ]
        
        # 按权重选择区域
        region_probs = [r[3] for r in regions]
        selected = self.rng.choice(regions, p=region_probs)
        name, reg_min, reg_max, _ = selected
        reg_min = float(reg_min)
        reg_max = float(reg_max)
        
        # 动态长度生成
        length_ranges = {
            'low': (20.0, 60.0),
            'mid': (40.0, 80.0),
            'high': (30.0, 70.0)
        }
        min_len, max_len = length_ranges[name]
        sub_length = self.rng.uniform(min_len, max_len)
        
        # 计算起始位置
        max_start = reg_max - sub_length
        start = self.rng.uniform(
            float(reg_min), 
            max(float(reg_min), max_start))
        
        # 边界保护
        start = np.clip(start, 5.0, 200.0 - sub_length)
        end = start + sub_length
        sub_range = (round(start, 2), round(end, 2))  # ✅ 新增此行
        print(f"Generated sub_range for material {material_idx}: {sub_range}")  # ✅ 修正
        self.rng.bit_generator.state = original_state
        return sub_range

    def _size_distribution(self, n, sub_range, material_idx):
        """改进的尺寸分布生成（自适应偏态）"""
        min_size, max_size = sub_range
        range_span = max_size - min_size
        
        # 动态参数生成
        mu = self.rng.uniform(
            min_size + 0.3*(max_size - min_size),  # 原0.2 → 0.3
            min_size + 0.7*(max_size - min_size)    # 原0.8 → 0.7
        )
        sigma = self.rng.uniform(0.05*(max_size - min_size), 0.15*(max_size - min_size))
        
        # 70%概率接近正态，30%显著偏态
        if self.rng.random() < 0.7:
            skew = self.rng.uniform(-0.8, 0.8)
        else:
            skew = self.rng.choice([-1.5, 1.5]) * self.rng.beta(2,5)
        
        # 生成偏态分布
        samples = skewnorm.rvs(
            a=skew,
            loc=mu,
            scale=sigma,
            size=n,
            random_state=self.rng
        )
        
        # 二次调整确保在区间内
        samples = np.clip(samples, min_size, max_size)
        jitter = self.rng.normal(0, 0.005*range_span, size=n)
        return np.round(samples + jitter, 2)

    def _strain_distribution(self, n):
        return np.clip(
            self.rng.normal(0.003, 0.001, size=n),
            *self.config.param_ranges['strain']
        )

    def _split_prob_distribution(self, n):
        return self.rng.beta(1.5, 5, size=n)

    def _noise_distribution(self):
        return self.rng.uniform(*self.config.param_ranges['noise_level'])

    # ----------------------
    # 新增数据保存方法
    # ----------------------
    def _save_plot_data(self, theta, original, simulated, params, filename):
        """筛选并保存5-90度数据"""
        mask = (theta >= 5) & (theta <= 90)
        filtered_data = {
            'theta': theta[mask],
            'Original': original[mask],
            'Simulated': simulated[mask]
        }
        
        data_dir = os.path.join(self.config.plot_folder, "plot_data")
        os.makedirs(data_dir, exist_ok=True)
        base_path = os.path.join(data_dir, os.path.splitext(filename)[0])

        # 保存筛选后的主数据
        pd.DataFrame(filtered_data).to_csv(f"{base_path}_main.csv", index=False)

        # 2. 保存晶粒尺寸分布
        pd.DataFrame({
            'Crystallite_Size': params['crystallite_sizes']
        }).to_csv(f"{base_path}_sizes.csv", index=False)

        # 3. 保存FWHM数据
        fwhm_values = self._calculate_fwhm(theta, 
                                         params['crystallite_sizes'].mean(),
                                         params['strains'].mean())
        pd.DataFrame({
            '2θ': theta,
            'FWHM': fwhm_values
        }).to_csv(f"{base_path}_fwhm.csv", index=False)

        # 4. 保存残差数据
        residuals = simulated - original
        pd.DataFrame({
            'Residual': residuals
        }).to_csv(f"{base_path}_residuals.csv", index=False)

        # 5. 保存元数据
        with open(f"{base_path}_metadata.json", 'w') as f:
            json.dump({
                'global_shift': params['global_shift'],
                'noise_level': params['noise_level'],
                'creation_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                'crystallite_size_mean': float(np.mean(params['crystallite_sizes'])),
                'strain_mean': float(np.mean(params['strains']))
            }, f, indent=2)

    # ----------------------
    # 物理模型核心改进
    # ----------------------
    @staticmethod
    def _voigt_profile(x, center, fwhm_g, fwhm_l):
        """优化的Voigt函数"""
        # 计算高斯部分的标准差（sigma_g）和洛伦兹部分的半宽度（gamma）
        sigma_g = fwhm_g / (2 * np.sqrt(2 * np.log(2)))
        gamma = fwhm_l / 2

        # 确保 x 和 center 具有相同的形状
        center = np.broadcast_to(center, x.shape)
        gamma = np.broadcast_to(gamma, x.shape)
        sigma_g = np.broadcast_to(sigma_g, x.shape)

        # 计算 Voigt 函数
        z = (x - center + 1j * gamma) / (sigma_g * np.sqrt(2))
        
        return np.real(special.wofz(z)) / (sigma_g * np.sqrt(2 * np.pi))

    def _calculate_fwhm(self, theta, size, strain):
        """改进的FWHM计算（处理单个theta值）"""
        theta_rad = np.deg2rad(theta / 2)
        
        # 样品展宽
        size_term = 0.9 * self.config.wavelength / (size * np.cos(theta_rad))
        strain_term = 4 * strain * np.tan(theta_rad)
        H_sample = np.sqrt(size_term**2 + strain_term**2)
        
        # 仪器展宽
        H_instrument = np.sqrt(
            self.config.U * np.tan(theta_rad)**2 +
            self.config.V * np.tan(theta_rad) +
            self.config.W
        )
        
        return np.sqrt(H_sample**2 + H_instrument**2)

    def _thermal_displacement(self, intensity, theta):
        """精确的热漫散射计算"""
        theta_rad = np.deg2rad(theta)
        q = 4 * np.pi * np.sin(theta_rad/2) / self.config.wavelength  # 散射矢量 (nm⁻¹)
        dw_factor = np.exp(-0.5 * (q**2) * self.config.atomic_displacement * 1e-2)  # 单位转换
        return intensity * dw_factor

    # ----------------------
    # 数据处理流程增强
    # ----------------------
    def _detect_peaks(self, theta, intensity):
        """自适应峰检测"""
        dynamic_range = np.percentile(intensity, 99) - np.percentile(intensity, 10)
        min_height = np.percentile(intensity, 85) * 0.5
        min_prominence = dynamic_range * 0.15
        
        peaks, props = signal.find_peaks(
            intensity,
            height=min_height,
            prominence=min_prominence,
            width=(3, 30),
            distance=10
        )
        
        if len(peaks) == 0:
            return np.array([]), np.array([])
            
        peak_scores = props['prominences'] * props['widths']
        selected = peaks[np.argsort(peak_scores)[-self.config.top_peaks:]]
        return theta[selected], intensity[selected]

    def _generate_parameters(self, theta_centers, filename):
        """参数生成改进"""
        num_peaks = len(theta_centers)
        
        filepath_hash = hash(os.path.abspath(filename))  # 绝对路径更唯一
        material_id = filepath_hash % 1000000  # 模数增大到 1000000
        
        # 生成子区间
        material_sub_range = self._generate_sub_range(material_idx=material_id)
        
        # 其他参数生成
        return {
        'global_shift': self.rng.uniform(*self.config.param_ranges['global_shift']),
        'crystallite_sizes': [
            self._size_distribution(1, material_sub_range, material_id).item()
            for _ in range(num_peaks)
        ],
            'strains': self.distributions['strain'](num_peaks),
            'split_probs': self.distributions['split_prob'](num_peaks),
            'preferred_factors': np.where(
                self.rng.random(num_peaks) < 0.15,
                self.rng.uniform(*self.config.param_ranges['preferred_factor'], num_peaks),
                1.0
            ),
            'noise_level': self.distributions['noise'](),
            'sub_range': material_sub_range,
            'material_id': material_id
        }

    def _simulate_pattern(self, theta, centers, intensities, params):
        """增强的模拟流程（移除错误归一化）"""
        simulated = np.zeros_like(theta)
        shifted_centers = centers + params['global_shift']
        
        for i, center in enumerate(shifted_centers):
            # 参数约束
            params['crystallite_sizes'][i] = np.clip(
                params['crystallite_sizes'][i],
                params['sub_range'][0],
                params['sub_range'][1]
            )
            
            # 获取当前峰参数
            size = params['crystallite_sizes'][i]
            strain = params['strains'][i]
            
            # 计算展宽
            fwhm_g = self._calculate_fwhm(center, size, strain)
            fwhm_l = fwhm_g * 0.5
            
            # 主峰生成
            main_peak = self._voigt_profile(theta, center, fwhm_g, fwhm_l)
            main_peak *= intensities[i] * params['preferred_factors'][i]
            simulated += main_peak
            
            # 峰分裂（概率性）
            if params['split_probs'][i] > 0.3:
                offset = self.rng.uniform(*self.config.param_ranges['split_angle'])
                for sign in [-1, 1]:
                    split_center = center + sign * offset
                    split_fwhm = fwhm_g * 1.2
                    split_peak = self._voigt_profile(theta, split_center,
                                                split_fwhm,
                                                split_fwhm*0.6)
                    simulated += split_peak * intensities[i] * 0.2
                    
            # 热漫散射
            thermal_intensity = self._thermal_displacement(intensities[i], center)
            thermal_peak = self._voigt_profile(theta, center,
                                            fwhm_g * 2,
                                            fwhm_g * 0.3)
            simulated += thermal_peak * thermal_intensity * 0.1
        
        # 添加噪声（基于原始峰高）
        max_peak = np.max(intensities) if len(intensities) > 0 else 1.0
        noise = self.rng.normal(0, params['noise_level'] * max_peak, len(theta))
        simulated += noise
        
        return simulated  # 保持原始数值范围

    # ----------------------
    # 可视化与输出
    # ----------------------
    def _create_diagnostic_plot(self, theta, original, simulated, params, filename):
        """改进的对比图（去除非必要元素）"""
        # 数据筛选
        mask = (theta >= 5) & (theta <= 90)
        theta = theta[mask]
        original = original[mask]
        simulated = simulated[mask]
        
        plt.figure(figsize=(20, 10))
        
        # ----------------------
        # 主对比图（左上）
        # ----------------------
        plt.subplot(2,2,1)  # 修改布局为2x2
        plt.plot(theta, original, label='Original', alpha=0.7, linewidth=1.5)
        plt.plot(theta, simulated, label='Simulated', alpha=0.7, linewidth=1.5)
        plt.title(f"Global Shift: {params['global_shift']:.3f}°\nNoise Level: {params['noise_level']:.4f}")
        plt.xlabel('2θ (°)')
        plt.ylabel('Normalized Intensity')
        plt.legend()
        
        # ----------------------
        # 晶粒尺寸分布（右上）
        # ----------------------
        plt.subplot(2,2,2)
        sizes = params['crystallite_sizes']
        sub_min, sub_max = params['sub_range']
        print(f"Plotting sub_range for {filename}: {sub_min}-{sub_max}")  # 新增调试输出

        plt.hist(sizes, 
                bins=20, 
                density=True,
                range=(sub_min, sub_max),
                alpha=0.7, 
                color='#1f77b4',
                edgecolor='black',
                linewidth=0.7)
        
        plt.xlim(sub_min, sub_max)  # 新增：固定X轴范围
        plt.axvline(sub_min, color='red', linestyle='--', alpha=0.5)
        plt.axvline(sub_max, color='red', linestyle='--', alpha=0.5)
        plt.text(sub_min, plt.ylim()[1]*0.9, f'{sub_min:.1f}', ha='left', color='red')
        plt.text(sub_max, plt.ylim()[1]*0.9, f'{sub_max:.1f}', ha='right', color='red')
        plt.title(f"Grain Size Distribution\n[{sub_min:.1f}-{sub_max:.1f} nm]")
        plt.xlabel('Crystallite Size (nm)')
        plt.ylabel('Probability Density')
        plt.grid(True, linestyle='--', alpha=0.3)  # 添加辅助网格
        
        # ----------------------
        # FWHM分布（左下）
        # ----------------------
        plt.subplot(2,2,3)
        fwhm_values = self._calculate_fwhm(theta, 
                                        np.mean(params['crystallite_sizes']),  
                                        params['strains'].mean())             
        plt.plot(theta, fwhm_values, color='#2ca02c', linewidth=2)
        plt.title("FWHM Distribution")
        plt.xlabel('2θ (°)')
        plt.ylabel('FWHM')
        plt.fill_between(theta, fwhm_values, alpha=0.2, color='#2ca02c')  # 添加填充
        
        # ----------------------
        # 残差分布（右下）
        # ----------------------
        plt.subplot(2,2,4)
        residuals = simulated - original
        plt.hist(residuals, 
                bins=50, 
                alpha=0.7, 
                color='#d62728',  # 使用标准红色
                edgecolor='black')
        plt.title(f"Residual Distribution\n(σ={np.std(residuals):.4f})")
        plt.xlabel('Residual Value')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plot_path = os.path.join(self.config.plot_folder, f"{filename}_diagnostic.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _estimate_beta_params(self, data, data_min, data_max):
        """估计Beta分布参数（用于可视化）"""
        # 归一化数据到[0,1]
        norm_data = (data - data_min) / (data_max - data_min)
        mean = np.mean(norm_data)
        var = np.var(norm_data)
        
        # 矩估计
        alpha = mean * (mean*(1-mean)/var - 1)
        beta = (1-mean) * (mean*(1-mean)/var - 1)
        return max(alpha, 0.1), max(beta, 0.1)  # 防止负值

    def _process_file(self, filepath, gen_plot=False):
        try:
            df = pd.read_excel(filepath, engine='openpyxl')
            theta = df['theta'].values.astype(float)
            raw_intensity = df['intensity'].values.astype(float)
            
            # 原始数据归一化（0-1）
            if np.ptp(raw_intensity) > 1e-6:  # 处理非零数据
                intensity = (raw_intensity - np.min(raw_intensity)) / np.ptp(raw_intensity)
            else:
                intensity = raw_intensity  # 全零数据保持原样
                
            # 峰值检测
            centers, intensities = self._detect_peaks(theta, intensity)
            if len(centers) == 0:
                logging.warning(f"No peaks in {os.path.basename(filepath)}")
                return
                
            # 参数生成与模拟
            params = self._generate_parameters(centers, filepath)
            raw_simulated = self._simulate_pattern(theta, centers, intensities, params)
            
            # 最终归一化（确保0-1）
            if np.ptp(raw_simulated) > 1e-6:
                simulated_normalized = (raw_simulated - np.min(raw_simulated)) / np.ptp(raw_simulated)
                simulated_normalized = np.clip(simulated_normalized, 0.0, 1.0)  # 严格限制范围
            else:
                simulated_normalized = raw_simulated  # 全零数据保持原样
            
            # 保存数据
            output_df = pd.DataFrame({
                'theta': theta,
                'intensity': simulated_normalized,
                'chemical_formula': df['chemical_formula']
            })
            output_path = os.path.join(self.config.output_folder, 
                                    f"enhanced_{os.path.basename(filepath)}")
            output_df.to_excel(output_path, index=False)
            
            # 生成诊断图（使用归一化后的数据）
            if gen_plot:
                self._create_diagnostic_plot(theta, intensity, simulated_normalized, params,
                                        os.path.splitext(os.path.basename(filepath))[0])
                
        except Exception as e:
            logging.error(f"Error processing {filepath}: {str(e)}")
            raise

    # ----------------------
    # 并行执行控制
    # ----------------------
    def execute(self):
        """优化的执行流程"""
        all_files = [os.path.join(self.config.input_folder, f) 
                    for f in sorted(os.listdir(self.config.input_folder))
                    if f.endswith('.xlsx')]
        
        with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
            # 预览阶段
            if self.config.demo_num > 0:
                demo_files = all_files[:self.config.demo_num]
                futures = [executor.submit(self._process_file, f, True) for f in demo_files]
                for f in tqdm(futures, desc="Preview Generation"):
                    f.result()
                    
                if not self._user_confirmation():
                    return
                
            # 批量处理
            remaining = all_files[self.config.demo_num:]
            if remaining:
                futures = {executor.submit(self._process_file, f, False): f for f in remaining}
                with tqdm(total=len(remaining), desc="Batch Processing") as pbar:
                    for future in futures:
                        future.result()
                        pbar.update(1)

    def _user_confirmation(self):
        """用户确认"""
        print(f"\nDiagnostic plots saved to: {self.config.plot_folder}")
        while True:
            choice = input("Proceed with full processing? [y/n]: ").lower()
            if choice == 'y': return True
            if choice == 'n': exit(0)
            print("Invalid input. Please enter y/n.")

# ----------------------
# 执行入口
# ----------------------
if __name__ == "__main__":
    config = EnhancedSimulationConfig()
        # ========== 验证代码 ==========
    test_simulator = EnhancedXRDSimulator(config)
    print("\n子区间生成验证（10个示例）：")
    for i in range(10):
        sub_range = test_simulator._generate_sub_range(material_idx=i)
        print(f"样本{i+1:02d}: {sub_range[0]:>5.1f} nm - {sub_range[1]:>5.1f} nm")
    simulator = EnhancedXRDSimulator(config)
    simulator.execute()