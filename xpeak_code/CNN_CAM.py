import pandas as pd
import numpy as np
import json
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

# 设置基础路径
base_path = ""
save_path = os.path.join(base_path, "CNN_Results")
plot_data_path = os.path.join(save_path, "plot_data")

print("使用CPU单进程处理，避免重复加载模型")


def plot_training_history(history, save_path):
    """绘制训练历史 - 只显示训练集"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 设置字体和边框样式
    title_font = {'size': 18, 'fontweight': 'bold'}
    label_font = {'size': 16, 'fontweight': 'bold'}
    tick_font = {'size': 14, 'fontweight': 'bold'}

    # 设置边框加粗
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_linewidth(2)

    # 损失曲线
    ax1.plot(history['loss'], label='Training Loss', linewidth=3, color='blue')
    ax1.set_title('Model Loss', fontdict=title_font)
    ax1.set_xlabel('Epoch', fontdict=label_font)
    ax1.set_ylabel('Loss', fontdict=label_font)
    ax1.legend(fontsize=14, loc='upper right')
    ax1.grid(True)

    # 设置刻度字体
    ax1.tick_params(axis='both', which='major', labelsize=14, width=2)
    ax1.tick_params(axis='both', which='minor', labelsize=12, width=2)

    # RMSE曲线
    if 'rmse' in history:
        ax2.plot(history['rmse'], label='Training RMSE', linewidth=3, color='red')
        ax2.set_title('RMSE', fontdict=title_font)
        ax2.set_xlabel('Epoch', fontdict=label_font)
        ax2.set_ylabel('RMSE', fontdict=label_font)
        ax2.legend(fontsize=14, loc='upper right')
        ax2.grid(True)

        # 设置刻度字体
        ax2.tick_params(axis='both', which='major', labelsize=14, width=2)
        ax2.tick_params(axis='both', which='minor', labelsize=12, width=2)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_history.png'), dpi=480, bbox_inches='tight')
    plt.show()


def plot_predictions(y_true, y_pred, title, catalyst_name, save_path):
    """绘制预测结果对比图"""
    plt.figure(figsize=(10, 6))

    # 设置字体和边框样式
    title_font = {'size': 18, 'fontweight': 'bold'}
    label_font = {'size': 16, 'fontweight': 'bold'}
    tick_font = {'size': 14, 'fontweight': 'bold'}

    # 计算R²分数
    r2 = r2_score(y_true, y_pred)

    # 绘制散点图
    plt.scatter(y_true, y_pred, alpha=0.6, s=50)

    # 绘制理想预测线
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='Ideal Prediction')

    plt.xlabel('True Values', fontdict=label_font)
    plt.ylabel('Predictions', fontdict=label_font)

    # 使用催化剂名称作为标题的一部分
    if catalyst_name:
        plt.title(f'{title} - {catalyst_name}\nR² = {r2:.4f}', fontdict=title_font)
    else:
        plt.title(f'{title}\nR² = {r2:.4f}', fontdict=title_font)

    plt.legend(fontsize=14, loc='upper left')  # 加大图例
    plt.grid(True, alpha=0.3)

    # 设置边框加粗和刻度字体
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(axis='both', which='major', labelsize=14, width=2)  # 加大刻度数字
    ax.tick_params(axis='both', which='minor', labelsize=12, width=2)

    # 保存图片
    filename = f"{title.lower().replace(' ', '_')}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    plt.show()


def plot_single_cam_curve(model, X_sample, feature_names, sample_idx, catalyst_name, save_path, dataset_type):
    """
    绘制单个样本的类激活图和特征曲线
    """
    try:
        # 获取XRD特征索引
        with open(os.path.join(plot_data_path, "xrd_feature_indices.json"), 'r') as f:
            xrd_indices = json.load(f)
        with open(os.path.join(plot_data_path, "xrd_feature_names.json"), 'r') as f:
            xrd_feature_names = json.load(f)

        if len(xrd_indices) == 0:
            print("警告：未找到XRD特征，将使用所有特征")
            xrd_indices = list(range(len(feature_names)))
            xrd_feature_names = feature_names

        # 提取XRD特征
        X_sample_xrd = X_sample[xrd_indices]

        # 创建Grad-CAM模型
        grad_model = Model(
            inputs=model.input,
            outputs=[model.get_layer('conv1d_5').output, model.output]
        )

        # 计算梯度
        with tensorflow.GradientTape() as tape:
            conv_outputs, predictions = grad_model(X_sample[np.newaxis, :])
            loss = predictions[:, 0]

        # 计算梯度
        grads = tape.gradient(loss, conv_outputs)

        # 全局平均池化梯度
        pooled_grads = tensorflow.reduce_mean(grads, axis=(0, 1))

        # 将特征图与权重相乘
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tensorflow.newaxis]
        heatmap = tensorflow.squeeze(heatmap)

        # 应用ReLU并归一化
        heatmap = tensorflow.maximum(heatmap, 0)
        heatmap /= tensorflow.math.reduce_max(heatmap)

        heatmap = heatmap.numpy()

        # 由于CNN的下采样，我们需要对热力图进行上采样以匹配原始特征维度
        from scipy import interpolate
        x_original = np.arange(len(heatmap))
        x_new = np.linspace(0, len(heatmap) - 1, X_sample.shape[0])

        f = interpolate.interp1d(x_original, heatmap, kind='linear', fill_value='extrapolate')
        heatmap_resized = f(x_new)

        # 提取XRD特征对应的热力图
        heatmap_xrd = heatmap_resized[xrd_indices]

        # 创建合并图
        fig, ax1 = plt.subplots(figsize=(15, 8))

        # 设置字体和边框样式
        title_font = {'size': 20, 'fontweight': 'bold'}
        label_font = {'size': 18, 'fontweight': 'bold'}
        tick_font = {'size': 16, 'fontweight': 'bold'}

        # 创建特征索引
        x_indices = np.arange(len(X_sample_xrd))

        # 绘制类激活图作为背景衬底（黄红渐变）
        y_min, y_max = X_sample_xrd.min() - 0.5, X_sample_xrd.max() + 0.5
        heatmap_background = np.tile(heatmap_xrd, (100, 1))

        # 绘制背景热力图
        im = ax1.imshow(heatmap_background, extent=[0, len(X_sample_xrd) - 1, y_min, y_max],
                        aspect='auto', cmap='YlOrRd', alpha=0.6, origin='lower')

        # 绘制特征曲线 - 钢蓝色
        steel_blue = '#4682B4'
        ax1.plot(x_indices, X_sample_xrd, color=steel_blue, linewidth=3, label='Intensity')

        # 设置x轴标签为衍射角值
        try:
            # 将特征名称转换为浮点数（衍射角）
            angles = np.array([float(name) for name in xrd_feature_names])

            # 设置主刻度（每10度）
            major_ticks = np.arange(10, 91, 10)
            major_tick_indices = []
            major_tick_labels = []
            for tick in major_ticks:
                idx = np.argmin(np.abs(angles - tick))
                if idx < len(angles):
                    major_tick_indices.append(idx)
                    major_tick_labels.append(f'{angles[idx]:.0f}')

            # 设置次刻度（每5度）
            minor_ticks = np.arange(5, 91, 5)
            minor_tick_indices = []
            for tick in minor_ticks:
                idx = np.argmin(np.abs(angles - tick))
                if idx < len(angles):
                    minor_tick_indices.append(idx)

            ax1.set_xticks(major_tick_indices)
            ax1.set_xticklabels(major_tick_labels, fontsize=16, fontweight='bold')
            ax1.set_xticks(minor_tick_indices, minor=True)

        except ValueError:
            # 如果特征名称不是数字，回退到每隔20个特征显示一个标签
            xtick_positions = np.arange(0, len(X_sample_xrd), 20)
            xtick_labels = [f'{i}' for i in xtick_positions]
            ax1.set_xticks(xtick_positions)
            ax1.set_xticklabels(xtick_labels, fontsize=16, fontweight='bold')

        ax1.set_xlabel('2θ', fontdict=label_font)
        ax1.set_ylabel('Intensity', fontdict=label_font)

        # 去掉纵坐标刻度
        ax1.tick_params(axis='y', which='both', left=False, labelleft=False)

        # 使用催化剂名称和数据集类型作为标题
        if catalyst_name:
            ax1.set_title(f'{catalyst_name} ({dataset_type.upper()} Set)', fontdict=title_font)
        else:
            ax1.set_title(f'Sample {sample_idx} ({dataset_type.upper()} Set)', fontdict=title_font)

        # 加大加粗图例
        ax1.legend(fontsize=16, loc='upper right', frameon=True, edgecolor='black')
        ax1.grid(True, alpha=0.3)
        ax1.grid(True, which='minor', alpha=0.2)
        ax1.set_xlim(0, len(X_sample_xrd) - 1)

        # 设置边框加粗和刻度字体
        for spine in ax1.spines.values():
            spine.set_linewidth(3)
        ax1.tick_params(axis='both', which='major', labelsize=16, width=3)
        ax1.tick_params(axis='both', which='minor', labelsize=14, width=2)

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Activation Intensity', fontdict=label_font)
        cbar.ax.tick_params(labelsize=14, width=2)
        cbar.ax.yaxis.label.set_fontweight('bold')
        cbar.ax.yaxis.label.set_fontsize(16)

        plt.tight_layout()

        # 根据数据集类型保存文件
        filename = f'xrd_cam_curve_{dataset_type}_sample_{sample_idx}.png'
        plt.savefig(os.path.join(save_path, filename), dpi=480, bbox_inches='tight')
        plt.close(fig)  # 关闭图形，避免内存泄漏

        # 保存热力图数据
        cam_data = {
            'xrd_feature_names': xrd_feature_names,
            'xrd_feature_values': X_sample_xrd.tolist(),
            'activation_heatmap': heatmap_xrd.tolist(),
            'catalyst_name': catalyst_name,
            'dataset_type': dataset_type,
            'excluded_features': list(set(feature_names) - set(xrd_feature_names))
        }

        with open(os.path.join(save_path, f'xrd_cam_data_{dataset_type}_sample_{sample_idx}.json'), 'w') as f:
            json.dump(cam_data, f, indent=4)

        print(f"已完成: {dataset_type}集样本 {sample_idx} - {catalyst_name}")

        return True

    except Exception as e:
        print(f"生成XRD特征CAM图时出错 (样本 {sample_idx}): {e}")
        return False


def generate_cam_for_dataset_batch(model, X_data, feature_names, catalyst_names_data, dataset_type, save_path,
                                   batch_size=10, max_samples=None):
    """
    使用批处理生成CAM图，避免内存问题
    """
    print(f"\n生成{dataset_type.upper()}集的XRD特征类激活图...")
    print(f"使用批处理，批次大小: {batch_size}")

    # 如果没有指定最大样本数，则使用所有样本
    if max_samples is None:
        max_samples = len(X_data)

    # 限制生成的样本数量
    num_samples = min(len(X_data), max_samples)
    print(f"计划生成 {num_samples} 个样本的CAM图")

    completed = 0
    successful = 0

    # 分批处理
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_size_current = batch_end - batch_start

        print(f"\n处理批次 {batch_start // batch_size + 1}/{(num_samples - 1) // batch_size + 1}: "
              f"样本 {batch_start} 到 {batch_end - 1}")

        for i in range(batch_start, batch_end):
            catalyst_name = catalyst_names_data[i] if i < len(catalyst_names_data) else f"Sample_{i}"

            print(f"生成{dataset_type.upper()}集样本 {i + 1}/{num_samples} 的XRD特征CAM图...")

            success = plot_single_cam_curve(
                model, X_data[i], feature_names, i, catalyst_name, save_path, dataset_type
            )

            completed += 1
            if success:
                successful += 1

            # 显示进度
            progress = (completed / num_samples) * 100
            print(
                f"进度: {completed}/{num_samples} ({progress:.1f}%) - 成功: {successful}, 失败: {completed - successful}")

        # 批次完成后清理内存
        import gc
        gc.collect()
        print(f"批次完成，清理内存...")

    print(f"{dataset_type.upper()}集CAM图生成完成: {successful}/{num_samples} 成功")


def generate_cam_for_dataset_single(model, X_data, feature_names, catalyst_names_data, dataset_type, save_path,
                                    max_samples=None):
    """
    单进程生成CAM图（逐个样本处理）
    """
    print(f"\n生成{dataset_type.upper()}集的XRD特征类激活图...")
    print("使用单样本处理模式")

    # 如果没有指定最大样本数，则使用所有样本
    if max_samples is None:
        max_samples = len(X_data)

    # 限制生成的样本数量
    num_samples = min(len(X_data), max_samples)

    completed = 0
    successful = 0

    for i in range(num_samples):
        catalyst_name = catalyst_names_data[i] if i < len(catalyst_names_data) else f"Sample_{i}"

        print(f"生成{dataset_type.upper()}集样本 {i + 1}/{num_samples} 的XRD特征CAM图...")

        success = plot_single_cam_curve(
            model, X_data[i], feature_names, i, catalyst_name, save_path, dataset_type
        )

        completed += 1
        if success:
            successful += 1

        # 显示进度
        progress = (completed / num_samples) * 100
        print(f"进度: {completed}/{num_samples} ({progress:.1f}%) - 成功: {successful}, 失败: {completed - successful}")

        # 每处理10个样本清理一次内存
        if (i + 1) % 10 == 0:
            import gc
            gc.collect()
            print("内存清理...")

    print(f"{dataset_type.upper()}集CAM图生成完成: {successful}/{num_samples} 成功")


def main():
    # 检查数据文件是否存在
    if not os.path.exists(plot_data_path):
        print(f"错误：绘图数据目录不存在: {plot_data_path}")
        print("请先运行 train_model.py 来生成训练数据和模型")
        return

    # 加载样本数量信息
    with open(os.path.join(plot_data_path, "sample_info.json"), 'r') as f:
        sample_info = json.load(f)

    train_samples_count = sample_info['train_samples_count']
    test_samples_count = sample_info['test_samples_count']

    print(f"训练集样本数: {train_samples_count}")
    print(f"测试集样本数: {test_samples_count}")
    print(f"总样本数: {train_samples_count + test_samples_count}")

    # 加载模型路径
    model_path = os.path.join(save_path, 'cnn_model.h5')
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在: {model_path}")
        return

    # 在主进程中加载一次模型
    print("加载模型...")
    model = load_model(model_path)
    print("模型加载完成")

    # 加载数据
    print("加载绘图数据...")

    # 加载特征数据
    X_train = np.load(os.path.join(plot_data_path, "X_train.npy"))
    X_test = np.load(os.path.join(plot_data_path, "X_test.npy"))
    y_train = np.load(os.path.join(plot_data_path, "y_train.npy"))
    y_test = np.load(os.path.join(plot_data_path, "y_test.npy"))
    y_train_pred = np.load(os.path.join(plot_data_path, "y_train_pred.npy"))
    y_test_pred = np.load(os.path.join(plot_data_path, "y_test_pred.npy"))

    # 加载其他数据
    with open(os.path.join(plot_data_path, "catalyst_names_train.json"), 'r') as f:
        catalyst_names_train = json.load(f)
    with open(os.path.join(plot_data_path, "catalyst_names_test.json"), 'r') as f:
        catalyst_names_test = json.load(f)
    with open(os.path.join(plot_data_path, "feature_names.json"), 'r') as f:
        feature_names = json.load(f)
    with open(os.path.join(plot_data_path, "training_history.json"), 'r') as f:
        training_history = json.load(f)
    with open(os.path.join(plot_data_path, "metrics.json"), 'r') as f:
        metrics = json.load(f)

    print("数据加载成功")
    print(f"训练集大小: {X_train.shape}，测试集大小: {X_test.shape}")

    # 绘制训练历史
    print("绘制训练历史...")
    plot_training_history(training_history, save_path)

    # 绘制预测结果
    print("绘制预测结果...")
    plot_predictions(y_train, y_train_pred, "Training Set Predictions", "All Catalysts", save_path)
    plot_predictions(y_test, y_test_pred, "Test Set Predictions", "All Catalysts", save_path)

    # 选择处理模式
    use_batch_processing = True  # 设置为False使用逐个样本处理
    batch_size = 5  # 批处理大小

    if use_batch_processing:
        # 使用批处理模式生成CAM图
        # 训练集：生成所有样本的CAM图
        generate_cam_for_dataset_batch(model, X_train, feature_names, catalyst_names_train, "train", save_path,
                                       batch_size=batch_size, max_samples=None)

        # 测试集：生成所有样本的CAM图
        generate_cam_for_dataset_batch(model, X_test, feature_names, catalyst_names_test, "test", save_path,
                                       batch_size=batch_size, max_samples=None)
    else:
        # 使用逐个样本处理模式
        print("使用逐个样本处理模式生成CAM图")

        # 训练集：生成所有样本的CAM图
        generate_cam_for_dataset_single(model, X_train, feature_names, catalyst_names_train, "train", save_path,
                                        max_samples=None)

        # 测试集：生成所有样本的CAM图
        generate_cam_for_dataset_single(model, X_test, feature_names, catalyst_names_test, "test", save_path,
                                        max_samples=None)

    # 保存评估结果
    results_df = pd.DataFrame({
        'True_Value': y_test,
        'Predicted_Value': y_test_pred,
        'Residual': y_test - y_test_pred,
        'Catalyst_Name': catalyst_names_test[:len(y_test)]
    })
    results_df.to_excel(os.path.join(save_path, 'cnn_predictions.xlsx'), index=False)

    print(f"所有图表已保存到: {save_path}")


if __name__ == "__main__":
    # 设置TensorFlow使用CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # 设置TensorFlow日志级别
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    main()