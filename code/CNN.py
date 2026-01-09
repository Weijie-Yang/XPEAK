import pandas as pd
import numpy as np
import warnings
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
import seaborn as sns
from tensorflow.keras import Model


# 配置GPU设置
def setup_gpu():
    """配置GPU设置"""
    # 检查是否有可用的GPU
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # 设置GPU内存增长，避免一次性占用所有内存
            for gpu in gpus:
                tensorflow.config.experimental.set_memory_growth(gpu, True)
            print(f"找到 {len(gpus)} 个GPU设备: {[gpu.name for gpu in gpus]}")

            # 设置使用第一个GPU
            if len(gpus) > 0:
                tensorflow.config.experimental.set_visible_devices(gpus[0], 'GPU')
                print(f"已配置使用GPU: {gpus[0].name}")
                return True
        except RuntimeError as e:
            print(f"GPU配置错误: {e}")
    else:
        print("未找到GPU设备，将使用CPU进行训练")
    return False


# 在设置随机种子之前调用GPU设置
has_gpu = setup_gpu()

# 设置随机种子保证可重复性
tensorflow.random.set_seed(42)
np.random.seed(42)

# 设置基础路径 - 修改为Linux路径
base_path = ""
data_file_path = os.path.join(base_path, "")

# 创建固定的新文件夹
save_path = os.path.join(base_path, "")
os.makedirs(save_path, exist_ok=True)
print(f"结果将保存到: {save_path}")


def create_cnn_model(input_shape):
    """创建CNN模型"""
    model = Sequential([
        # 输入层重塑
        tensorflow.keras.layers.Reshape((input_shape[0], 1), input_shape=(input_shape[0],)),

        # 第一个卷积块
        Conv1D(64, kernel_size=3, activation='relu', padding='same', name='conv1d_1'),
        BatchNormalization(name='batch_norm_1'),
        Conv1D(64, kernel_size=3, activation='relu', padding='same', name='conv1d_2'),
        BatchNormalization(name='batch_norm_2'),
        MaxPooling1D(pool_size=2, name='max_pool_1'),
        Dropout(0.2, name='dropout_1'),

        # 第二个卷积块
        Conv1D(128, kernel_size=3, activation='relu', padding='same', name='conv1d_3'),
        BatchNormalization(name='batch_norm_3'),
        Conv1D(128, kernel_size=3, activation='relu', padding='same', name='conv1d_4'),
        BatchNormalization(name='batch_norm_4'),
        MaxPooling1D(pool_size=2, name='max_pool_2'),
        Dropout(0.3, name='dropout_2'),

        # 第三个卷积块（如果特征维度足够大）
        Conv1D(256, kernel_size=3, activation='relu', padding='same', name='conv1d_5'),
        BatchNormalization(name='batch_norm_5'),
        MaxPooling1D(pool_size=2, name='max_pool_3'),
        Dropout(0.4, name='dropout_3'),

        # 展平后接全连接层
        Flatten(name='flatten'),
        Dense(512, activation='relu', name='dense_1'),
        BatchNormalization(name='batch_norm_6'),
        Dropout(0.5, name='dropout_4'),
        Dense(256, activation='relu', name='dense_2'),
        Dropout(0.4, name='dropout_5'),
        Dense(128, activation='relu', name='dense_3'),
        Dropout(0.3, name='dropout_6'),
        Dense(1, name='output')  # 输出层，回归任务
    ])

    return model


def get_xrd_feature_indices(feature_names):
    """获取XRD特征对应的索引（排除非XRD特征）"""
    xrd_feature_indices = []
    xrd_feature_names = []

    # 定义需要排除的特征
    excluded_from_cam = [
        "Catalysts_Mass_Fraction(wt%)",
        "heating_rate(℃/min)"
    ]

    for i, name in enumerate(feature_names):
        # 只保留可以转换为浮点数的特征名称（XRD角度特征）
        try:
            float(name)
            xrd_feature_indices.append(i)
            xrd_feature_names.append(name)
        except ValueError:
            # 如果不是数字，检查是否在排除列表中
            if name not in excluded_from_cam:
                xrd_feature_indices.append(i)
                xrd_feature_names.append(name)

    return xrd_feature_indices, xrd_feature_names


def convert_to_serializable(obj):
    """将NumPy数据类型转换为JSON可序列化的Python原生类型"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def save_training_data_for_plotting(save_path, X_train, X_test, y_train, y_test,
                                    catalyst_names_train, catalyst_names_test,
                                    feature_names, model, scaler, history):
    """保存绘图所需的数据"""

    # 创建数据保存目录
    plot_data_path = os.path.join(save_path, "plot_data")
    os.makedirs(plot_data_path, exist_ok=True)

    # 保存特征数据
    np.save(os.path.join(plot_data_path, "X_train.npy"), X_train)
    np.save(os.path.join(plot_data_path, "X_test.npy"), X_test)
    np.save(os.path.join(plot_data_path, "y_train.npy"), y_train)
    np.save(os.path.join(plot_data_path, "y_test.npy"), y_test)

    # 保存催化剂名称
    with open(os.path.join(plot_data_path, "catalyst_names_train.json"), 'w') as f:
        json.dump(
            convert_to_serializable(catalyst_names_train.tolist() if hasattr(catalyst_names_train, 'tolist') else list(catalyst_names_train)), f)
    with open(os.path.join(plot_data_path, "catalyst_names_test.json"), 'w') as f:
        json.dump(convert_to_serializable(catalyst_names_test.tolist() if hasattr(catalyst_names_test, 'tolist') else list(catalyst_names_test)), f)

    # 保存特征名称
    with open(os.path.join(plot_data_path, "feature_names.json"), 'w') as f:
        json.dump(convert_to_serializable(feature_names), f)

    # 保存训练历史 - 转换为可序列化的格式
    print("正在转换训练历史数据为JSON可序列化格式...")
    if history is not None:
        serializable_history = convert_to_serializable(history.history)
        with open(os.path.join(plot_data_path, "training_history.json"), 'w') as f:
            json.dump(serializable_history, f, indent=4)
    else:
        print("警告：训练历史为空，跳过保存")

    # 保存预测结果
    y_train_pred = model.predict(X_train, verbose=0).flatten()
    y_test_pred = model.predict(X_test, verbose=0).flatten()

    np.save(os.path.join(plot_data_path, "y_train_pred.npy"), y_train_pred)
    np.save(os.path.join(plot_data_path, "y_test_pred.npy"), y_test_pred)

    # 保存XRD特征索引
    xrd_indices, xrd_feature_names = get_xrd_feature_indices(feature_names)
    with open(os.path.join(plot_data_path, "xrd_feature_indices.json"), 'w') as f:
        json.dump(convert_to_serializable(xrd_indices), f)
    with open(os.path.join(plot_data_path, "xrd_feature_names.json"), 'w') as f:
        json.dump(convert_to_serializable(xrd_feature_names), f)

    # 保存模型摘要
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    with open(os.path.join(plot_data_path, "model_summary.txt"), 'w') as f:
        f.write("\n".join(model_summary))

    # 保存评估指标
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    metrics = {
        'train_mse': convert_to_serializable(train_mse),
        'train_rmse': convert_to_serializable(train_rmse),
        'train_mae': convert_to_serializable(train_mae),
        'train_r2': convert_to_serializable(train_r2),
        'test_mse': convert_to_serializable(test_mse),
        'test_rmse': convert_to_serializable(test_rmse),
        'test_mae': convert_to_serializable(test_mae),
        'test_r2': convert_to_serializable(test_r2)
    }

    with open(os.path.join(plot_data_path, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)

    # 保存样本数量信息
    sample_info = {
        'train_samples_count': convert_to_serializable(len(X_train)),
        'test_samples_count': convert_to_serializable(len(X_test)),
        'total_samples_count': convert_to_serializable(len(X_train) + len(X_test))
    }
    with open(os.path.join(plot_data_path, "sample_info.json"), 'w') as f:
        json.dump(sample_info, f, indent=4)

    print(f"绘图数据已保存到: {plot_data_path}")
    print(f"训练集样本数: {len(X_train)}")
    print(f"测试集样本数: {len(X_test)}")
    print(f"总样本数: {len(X_train) + len(X_test)}")


def main():
    # 检查GPU信息
    print("=" * 50)
    print("GPU信息检查:")
    print("=" * 50)

    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    gpu_info = "无GPU设备"
    if gpus:
        gpu_info = f"检测到 {len(gpus)} 个GPU设备:"
        for i, gpu in enumerate(gpus):
            gpu_info += f"\n  GPU {i}: {gpu.name}"
            try:
                from tensorflow.python.client import device_lib
                local_device_protos = device_lib.list_local_devices()
                for device in local_device_protos:
                    if device.device_type == 'GPU':
                        gpu_info += f"\n    设备类型: {device.device_type}"
                        gpu_info += f"\n    内存限制: {device.memory_limit}"
            except:
                pass
    else:
        gpu_info = "未检测到GPU设备，使用CPU训练"

    print(gpu_info)
    print("=" * 50)

    # 加载数据
    print("加载数据...")
    try:
        df = pd.read_excel(data_file_path)
        print(f"数据加载成功，数据形状: {df.shape}")
        print(f"数据列名: {list(df.columns)}")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 检查目标列是否存在
    if 'Temperature_peak(℃)' not in df.columns:
        print("错误：数据文件中没有找到 'Temperature_peak(℃)' 列")
        print(f"可用的列: {list(df.columns)}")
        return

    # 保存催化剂名称
    if 'catalysts_component' in df.columns:
        catalyst_names = df['catalysts_component'].values
    else:
        catalyst_names = [f"Sample_{i}" for i in range(len(df))]
        print("警告：数据文件中没有找到 'catalysts_component' 列，将使用样本编号作为标题")

    y = df['Temperature_peak(℃)'].values

    # 修改排除列：训练时包含Catalysts_Mass_Fraction和heating_rate
    excluded_columns = [
        "paper_number", "catalyst_number", "catalysts_component",
        "Temperature_peak(℃)", "ΔTemperature_peak"
    ]

    # 只排除实际存在的列
    excluded_columns = [col for col in excluded_columns if col in df.columns]
    X = df.drop(excluded_columns, axis=1)

    # 确保所有特征名为字符串类型
    feature_names = X.columns.astype(str).tolist()
    X.columns = feature_names

    # 打印数据信息用于调试
    print(f"特征数: {len(X.columns)}，样本数: {len(X)}")
    print(f"排除的特征列: {excluded_columns}")
    print(f"包含的特征: {feature_names}")
    print(f"目标变量统计: 均值={y.mean():.2f}, 标准差={y.std():.2f}, 范围=[{y.min():.2f}, {y.max():.2f}]")

    # 数据标准化
    print("数据标准化...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分数据集
    print("划分训练集和测试集...")
    X_train, X_test, y_train, y_test, catalyst_names_train, catalyst_names_test = train_test_split(
        X_scaled, y, catalyst_names, test_size=0.1, random_state=42, shuffle=True
    )

    print(f"训练集大小: {X_train.shape}，测试集大小: {X_test.shape}")

    # 创建CNN模型
    input_shape = (X_train.shape[1],)
    print(f"输入形状: {input_shape}")

    model = create_cnn_model(input_shape)

    # 编译模型
    model.compile(
        optimizer=Adam(learning_rate=0.002),
        loss='mse',
        metrics=['mae', tensorflow.keras.metrics.RootMeanSquaredError(name='rmse')]
    )

    print("模型结构摘要:")

    # 获取模型摘要字符串
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    model_summary_str = "\n".join(model_summary)
    print(model_summary_str)

    # 定义回调函数
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=50,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=40,
        min_lr=1e-7,
        verbose=1
    )

    # 训练模型
    print("开始训练模型...")
    history = model.fit(
        X_train, y_train,
        epochs=300,
        batch_size=24,
        validation_split=0.1,
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
        shuffle=True
    )

    # 模型评估
    print("\n模型评估结果:")

    # 训练集评估
    y_train_pred = model.predict(X_train, verbose=0).flatten()
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    print("训练集表现:")
    print(f"MSE: {train_mse:.4f}")
    print(f"RMSE: {train_rmse:.4f}")
    print(f"MAE: {train_mae:.4f}")
    print(f"R²: {train_r2:.4f}")

    # 测试集评估
    y_test_pred = model.predict(X_test, verbose=0).flatten()
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("\n测试集表现:")
    print(f"MSE: {test_mse:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"MAE: {test_mae:.4f}")
    print(f"R²: {test_r2:.4f}")

    # 保存模型
    model.save(os.path.join(save_path, 'cnn_model.h5'))
    print("模型已保存")

    # 保存绘图数据
    save_training_data_for_plotting(save_path, X_train, X_test, y_train, y_test,
                                    catalyst_names_train, catalyst_names_test,
                                    feature_names, model, scaler, history)

    print(f"所有训练数据和模型已保存到: {save_path}")


if __name__ == "__main__":
    main()