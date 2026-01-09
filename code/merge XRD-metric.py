import os
import pandas as pd
import time
from decimal import Decimal, ROUND_HALF_UP
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import logging
import gc
from functools import partial  # 添加缺失的导入

# 配置路径参数
input_folder = ""
output_folder = ""
os.makedirs(output_folder, exist_ok=True)

# 配置日志记录
logging.basicConfig(
    filename=os.path.join(output_folder, 'processing.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def format_duration(seconds):
    """将秒转换为易读的时间格式"""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:05.2f}"

def precise_round(value):
    """统一保留两位小数（包含末尾零）"""
    return f"{Decimal(str(value)).quantize(Decimal('0.00'), rounding=ROUND_HALF_UP):.2f}"

def process_theta(filepath):
    """并行任务1：角度值提取"""
    try:
        df = pd.read_csv(filepath, usecols=['theta'])
        theta_set = {precise_round(float(x)) for x in df['theta']}
        logging.info(f"文件 {os.path.basename(filepath)} 提取到 {len(theta_set)} 个角度值")
        return (filepath, theta_set)
    except Exception as e:
        logging.error(f"角度提取失败 {os.path.basename(filepath)}: {str(e)}")
        return (filepath, e)

def process_file(filepath, sorted_theta):
    """并行任务2：文件处理"""
    try:
        # 仅读取必要列以减少内存消耗
        df = pd.read_csv(filepath, usecols=['theta', 'intensity', 'chemical_formula'])
        formula = df['chemical_formula'].iloc[0]
        
        # 创建角度-强度映射字典以提高查找速度
        angle_intensity = {}
        for _, row in df.iterrows():
            theta = precise_round(float(row['theta']))
            # 仅保留与角度矩阵匹配的数据
            if theta in sorted_theta:
                angle_intensity[theta] = row['intensity']
        
        # 构建结果行
        row = {"chemical_formula": formula}
        for theta in sorted_theta:
            row[theta] = angle_intensity.get(theta)
        
        logging.info(f"成功处理 {os.path.basename(filepath)} ({len(angle_intensity)}/{len(df)} 数据点)")
        return row
    except Exception as e:
        logging.error(f"数据处理失败 {os.path.basename(filepath)}: {str(e)}")
        return (filepath, e)

class ProgressTracker:
    """实时进度跟踪器"""
    def __init__(self, total):
        self.start_time = time.time()
        self.total = total
        self.completed = 0
        self.speed_history = []
        
    def update(self, increment=1):
        self.completed += increment
        elapsed = time.time() - self.start_time
        current_speed = self.completed / elapsed
        
        # 计算平滑处理速度
        alpha = 0.2
        if self.speed_history:
            smooth_speed = alpha * current_speed + (1-alpha) * self.speed_history[-1]
        else:
            smooth_speed = current_speed
        self.speed_history.append(smooth_speed)
        
        # 计算剩余时间
        remaining = (self.total - self.completed) / smooth_speed if smooth_speed > 0 else 0
        
        # 获取内存使用情况
        mem_usage = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
        
        # 构建进度条和状态信息
        progress_bar = f"[{'■'*int(20*self.completed/self.total):<20}] {self.completed}/{self.total}"
        stats = (f"速度: {smooth_speed:.1f}/s | "
                 f"已用: {format_duration(elapsed)} | "
                 f"剩余: {format_duration(remaining)} | "
                 f"内存: {mem_usage:.1f}GB")
        
        print(f"\r{progress_bar} {stats}", end='', flush=True)

def main():
    os.makedirs(output_folder, exist_ok=True)
    
    # ================== 阶段1: 收集所有csv文件 ==================
    print("\n[1/3] 扫描文件结构...")
    selected_files = [
        os.path.join(input_folder, f) 
        for f in os.listdir(input_folder) 
        if f.endswith('.csv')
    ]
    total_files = len(selected_files)
    print(f"发现 {total_files} 个CSV文件")
    time.sleep(0.5)  # 给用户时间阅读信息

    # ================== 阶段2: 角度索引处理 ==================
    print(f"\n[2/3] 建立角度索引（处理{len(selected_files)}个文件）...")
    progress = ProgressTracker(len(selected_files))
    all_theta = set()
    theta_errors = []
    
    # 使用进程池处理角度提取任务
    with ProcessPoolExecutor(max_workers=psutil.cpu_count(logical=False)) as executor:  # 使用物理核心数
        futures = {executor.submit(process_theta, f): f for f in selected_files}
        for future in as_completed(futures):
            filepath = futures[future]
            result = future.result()
            if isinstance(result[1], Exception):
                theta_errors.append((filepath, result[1]))
            else:
                all_theta.update(result[1])
            progress.update()
    
    print()  # 换行
    if not all_theta:
        print("警告: 没有有效的角度数据提取，请检查文件。")
        return
    
    sorted_theta = sorted(all_theta, key=lambda x: float(x))
    print(f"角度矩阵维度: {len(sorted_theta)} 列 ({sorted_theta[0]}~{sorted_theta[-1]})")
    time.sleep(1)  # 给用户时间阅读信息

    # ================== 阶段3: 批量数据处理和写入 ==================
    print("\n[3/3] 处理全部数据并写入单个文件...")
    batch_progress = ProgressTracker(total_files)
    
    # 创建最终输出文件
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_filename = f"consolidated_results_{timestamp}.csv"
    output_path = os.path.join(output_folder, output_filename)
    
    # 准备列名
    columns = ["chemical_formula"] + sorted_theta
    
    # 分块处理并直接写入文件，避免内存溢出
    chunk_size = min(5000, total_files)  # 根据文件数量调整chunk大小
    chunks = [selected_files[i:i+chunk_size] for i in range(0, total_files, chunk_size)]
    
    # 打开文件进行写入
    with open(output_path, 'w', encoding='utf-8-sig') as f:
        # 写入标题行
        f.write(",".join(columns) + "\n")
        
        for chunk_idx, file_chunk in enumerate(chunks, 1):
            chunk_results = []
            data_errors = []
            
            # 并行处理当前chunk的文件
            with ProcessPoolExecutor(max_workers=psutil.cpu_count(logical=False)) as executor:
                task_func = partial(process_file, sorted_theta=sorted_theta)
                future_map = {executor.submit(task_func, f): f for f in file_chunk}
                
                for future in as_completed(future_map):
                    filepath = future_map[future]
                    try:
                        result = future.result()
                        if isinstance(result, dict):
                            chunk_results.append(result)
                        else:
                            data_errors.append((filepath, result[1]))
                    except Exception as e:
                        data_errors.append((filepath, e))
                    finally:
                        batch_progress.update()
            
            # 处理当前chunk的结果
            if chunk_results:
                # 写入CSV行
                for row in chunk_results:
                    # 构建CSV行
                    csv_line = [row["chemical_formula"]]
                    for theta in sorted_theta:
                        csv_line.append(str(row.get(theta, "")))
                    f.write(",".join(csv_line) + "\n")
                
                print(f"块 {chunk_idx} (包含 {len(file_chunk)} 个文件) 已写入磁盘")
            
            # 错误报告
            if data_errors:
                print(f"块 {chunk_idx} 中的错误文件（示例）:")
                for f, e in data_errors[:3]:
                    print(f"  {os.path.basename(f)}: {str(e)[:100]}")
            
            # 清理内存
            del chunk_results
            gc.collect()
    
    # 输出总结
    print("\n处理完成！结果摘要:")
    print(f"总文件数: {total_files}")
    print(f"成功处理: {total_files - len(theta_errors) - len(data_errors)}")
    print(f"角度提取错误: {len(theta_errors)}")
    print(f"数据处理错误: {len(data_errors)}")
    
    if theta_errors:
        print("\n角度提取错误文件（示例）:")
        for f, e in theta_errors[:3]:
            print(f"  {os.path.basename(f)}: {str(e)[:100]}")
    
    print(f"\n结果已保存到: {output_path}")

if __name__ == "__main__":
    main()