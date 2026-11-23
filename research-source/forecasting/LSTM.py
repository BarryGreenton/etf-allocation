
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 设置 matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def train_and_predict(data_file_path, current_dir):
    """
    对单个 ETF 文件进行训练和预测
    """
    # 提取 ETF 代码 (文件名)
    etf_code = os.path.basename(data_file_path).replace('.csv', '')
    print(f"\n{'='*20} 开始处理: {etf_code} {'='*20}")
    
    # 读取数据
    print(f"读取数据: {data_file_path}")
    df = pd.read_csv(data_file_path)
    
    # 确保日期列格式正确 (第1列，索引0)
    df['date'] = pd.to_datetime(df['date'])
    
    # 1. 数据预处理
    # 由于前250个交易日没有多因子得分这个特征，因此不做使用
    # 丢弃前250行
    df = df.iloc[250:].reset_index(drop=True)
    
    # 特征列选择
    # 第1列为交易日期 (index 0)
    # 第2列为收盘价 (index 1) -> 也是特征之一
    # 第19列为未来单日收益率 (index 18) -> 忽略
    # 第20列为多因子得分 (index 19) -> 特征
    # 所有特征列为：2-18列 (indices 1-17) 以及第20列 (index 19)
    feature_indices = list(range(1, 18)) + [19]
    feature_cols = df.columns[feature_indices].tolist()
    target_col_idx = 1 # 收盘价在 index 1
    
    print(f"使用特征列: {feature_cols}")
    print(f"特征数量: {len(feature_cols)}")
    
    # --- 新增逻辑：计算收盘价的每日变化 (t日收盘价 - t-1日收盘价) ---
    # 这将作为新的 Target
    # Target 是 close_diff
    close_price = df.iloc[:, target_col_idx]
    close_diff = close_price.diff()
    # 填充第一个 NaN 为 0
    close_diff = close_diff.fillna(0)
    
    # 划分训练集和测试集
    # 测试集为250天，剩余都是训练集
    test_days = 250
    train_size = len(df) - test_days
    
    if train_size <= 0:
        print(f"错误：数据量不足以划分测试集 ({etf_code})")
        return
        
    train_df = df.iloc[:train_size]
    # 注意：test_df 用于逻辑划分
    
    print(f"训练集大小: {len(train_df)} 天")
    print(f"测试集大小: {test_days} 天")
    
    # --- 标准化 ---
    
    # 1. 特征标准化 (X)
    scaler_X = StandardScaler()
    train_features = train_df.iloc[:, feature_indices].values
    scaler_X.fit(train_features)
    
    # 对整个数据集特征进行 transform
    all_features = df.iloc[:, feature_indices].values
    all_features_scaled = scaler_X.transform(all_features)
    
    # 2. 目标标准化 (y: close_diff)
    # 使用训练集的 diff 进行 fit
    scaler_y = StandardScaler()
    train_diff = close_diff.iloc[:train_size].values.reshape(-1, 1)
    scaler_y.fit(train_diff)
    
    # 对整个数据集的 diff 进行 transform
    all_diff_scaled = scaler_y.transform(close_diff.values.reshape(-1, 1))
    
    # 获取 diff 的反标准化参数
    diff_mean = scaler_y.mean_[0]
    diff_scale = scaler_y.scale_[0]
    
    # 准备 LSTM 数据
    # 回溯窗口定为30天
    sequence_length = 30
    
    def create_dataset(features_scaled, diff_scaled, raw_prices, start_idx, end_idx):
        """
        features_scaled: 标准化后的特征矩阵
        diff_scaled: 标准化后的收盘价变化量 (Target)
        raw_prices: 原始收盘价序列 (用于记录 t-1 日收盘价，以便还原)
        start_idx, end_idx: 预测目标的索引范围 [start_idx, end_idx)
        """
        X = []
        y = []
        base_prices = [] # t-1 日的真实收盘价
        
        for i in range(start_idx, end_idx):
            if i < sequence_length:
                continue
            
            # 输入特征：t-30 到 t-1 (长度 30)
            X.append(features_scaled[i-sequence_length : i])
            
            # 目标：t 日的收盘价变化量 (标准化后)
            y.append(diff_scaled[i, 0])
            
            # 基准价格：t-1 日的真实收盘价
            base_prices.append(raw_prices[i-1])
            
        return np.array(X), np.array(y), np.array(base_prices)
        
    # 获取原始收盘价数组
    raw_close_prices = df.iloc[:, target_col_idx].values
    
    # 构建训练集
    # 范围: 从 sequence_length 到 train_size
    print("构建训练集数据...")
    X_train, y_train, _ = create_dataset(all_features_scaled, all_diff_scaled, raw_close_prices, sequence_length, train_size)
    
    # 构建测试集
    # 范围: 从 train_size 到 len(df)
    print("构建测试集数据...")
    X_test, y_test_scaled, base_prices_test = create_dataset(all_features_scaled, all_diff_scaled, raw_close_prices, train_size, len(df))
    
    # 获取测试集真实的未归一化收盘价，用于最终对比
    test_dates = df.iloc[train_size:, 0].values
    y_test_real = df.iloc[train_size:, target_col_idx].values
    
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # 2. 模型构建与训练
    model = Sequential()
    # 输入形状: (30, 特征数量)
    model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1)) # 输出预测的 diff (scaled)
    
    # 损失函数选用MSE，优化器使用rmsprop
    model.compile(loss='mse', optimizer='rmsprop')
    
    print("开始训练模型...")
    # epochs=50, batch_size=32
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
    
    # 3. 保存模型
    model_save_dir = os.path.join(current_dir, 'model_checkpoints')
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
        
    model_path = os.path.join(model_save_dir, f'lstm_{etf_code}_diff.h5')
    model.save(model_path)
    print(f"模型已保存至: {model_path}")
    
    # 4. 预测与评估
    print("开始预测测试集...")
    # 预测得到标准化后的变化量
    pred_diff_scaled = model.predict(X_test)
    pred_diff_scaled = pred_diff_scaled.flatten()
    
    # 反标准化变化量
    # pred_diff = pred_diff_scaled * std + mean
    pred_diff = pred_diff_scaled * diff_scale + diff_mean
    
    # 计算预测收盘价
    # pred_price_t = true_price_{t-1} + pred_diff_t
    y_pred = base_prices_test + pred_diff
    
    # 计算 MSE (基于真实收盘价)
    mse = mean_squared_error(y_test_real, y_pred)
    print(f"测试集收盘价 MSE: {mse:.4f}")
    
    # --- 新增计算：收益率 ---
    # 真实日收益率 = (真实收盘价 - 上一日收盘价) / 上一日收盘价
    # 预测日收益率 = (预测收盘价 - 上一日收盘价) / 上一日收盘价 = 预测变化量 / 上一日收盘价
    
    # 注意：base_prices_test 就是上一日收盘价
    # 为了避免除以0 (虽然收盘价不太可能为0)，加个极小值或者不做处理(通常收盘价>0)
    
    true_returns = (y_test_real - base_prices_test) / base_prices_test
    pred_returns = pred_diff / base_prices_test
    
    # 计算日收益率的平方误差 (Squared Error)
    # 这里理解为每一天的预测误差的平方
    returns_se = (true_returns - pred_returns) ** 2
    
    # 整体收益率 MSE
    returns_mse = np.mean(returns_se)
    print(f"测试集日收益率 MSE: {returns_mse:.6f}")

    # 5. 保存测试结果
    result_save_dir = os.path.join(current_dir, 'test_results')
    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)
        
    # 保存 CSV
    result_df = pd.DataFrame({
        '交易日期': test_dates,
        '真实收盘价': y_test_real,
        '预测收盘价': y_pred,
        '上一日收盘价': base_prices_test,
        '预测变化量': pred_diff,
        '真实日收益率': true_returns,
        '预测日收益率': pred_returns,
        '收益率平方误差(SE)': returns_se
    })
    
    result_csv_path = os.path.join(result_save_dir, f'{etf_code}_test_predictions.csv')
    result_df.to_csv(result_csv_path, index=False, encoding='utf-8-sig')
    print(f"测试结果已保存至: {result_csv_path}")
    
    # 绘制折线图 1: 收盘价对比
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, y_test_real, label='真实收盘价', color='blue')
    plt.plot(test_dates, y_pred, label='预测收盘价', color='red', linestyle='--')
    
    plt.title(f'{etf_code} 收盘价预测对比 (预测变化量叠加)')
    plt.xlabel('日期')
    plt.ylabel('收盘价')
    plt.legend()
    plt.grid(True)
    
    # 格式化日期轴
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    
    plot_path = os.path.join(result_save_dir, f'{etf_code}_prediction_plot.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"收盘价对比图已保存至: {plot_path}")
    
    # 绘制折线图 2: 收益率对比
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, true_returns, label='真实日收益率', color='green', alpha=0.7)
    plt.plot(test_dates, pred_returns, label='预测日收益率', color='orange', linestyle='--', alpha=0.7)
    
    plt.title(f'{etf_code} 日收益率预测对比')
    plt.xlabel('日期')
    plt.ylabel('收益率')
    plt.legend()
    plt.grid(True)
    
    # 格式化日期轴
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    
    plot_path_returns = os.path.join(result_save_dir, f'{etf_code}_returns_plot.png')
    plt.savefig(plot_path_returns)
    plt.close()
    print(f"收益率对比图已保存至: {plot_path_returns}")

def main():
    # 路径设置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) # .../实验
    
    # 数据文件夹路径
    data_dir = os.path.join(project_root, '数据工程', '训练用数据')
    
    if not os.path.exists(data_dir):
        print(f"错误：找不到数据文件夹 {data_dir}")
        return
        
    # 获取所有 csv 文件
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("未找到任何 CSV 文件")
        return
        
    print(f"找到 {len(csv_files)} 个数据文件，准备开始训练...")
    
    for csv_file in csv_files:
        file_path = os.path.join(data_dir, csv_file)
        try:
            train_and_predict(file_path, current_dir)
        except Exception as e:
            print(f"处理文件 {csv_file} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
