
import os
import pandas as pd
import numpy as np

def calculate_covariance_matrices():
    # 路径设置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) # .../实验
    
    # 训练用数据文件夹路径
    data_dir = os.path.join(project_root, '数据工程', '训练用数据')
    
    # 结果保存文件夹路径
    save_dir = os.path.join(current_dir, 'covariance_matrices')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print(f"数据目录: {data_dir}")
    print(f"结果保存目录: {save_dir}")
    
    # 获取所有 csv 文件
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("未找到任何 CSV 文件")
        return
        
    print(f"找到 {len(csv_files)} 个数据文件")
    
    # 读取所有 ETF 数据
    etf_data = {}
    for f in csv_files:
        etf_code = f.replace('.csv', '')
        file_path = os.path.join(data_dir, f)
        try:
            # 尝试用 utf-8 读取
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            try:
                # 尝试用 gbk 读取
                df = pd.read_csv(file_path, encoding='gbk')
            except Exception as e:
                print(f"读取文件 {f} 失败: {str(e)}")
                continue
                
        # 确保日期列格式正确 (第1列，索引0)
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        # 将日期设置为索引
        df.set_index(df.columns[0], inplace=True)
        # 只保留收盘价 (第2列，索引0，因为日期变成了索引)
        # 原始表格第2列是收盘价，现在是第1列
        close_price = df.iloc[:, 0]
        close_price.name = etf_code
        etf_data[etf_code] = close_price
        
    # 合并所有 ETF 的收盘价到一个 DataFrame
    combined_df = pd.DataFrame(etf_data)
    
    # 按日期排序
    combined_df.sort_index(inplace=True)
    
    # 检查是否有缺失值并填充 (虽然预期数据应该是对齐的，但以防万一)
    combined_df.fillna(method='ffill', inplace=True)
    combined_df.dropna(inplace=True) # 删除最前面的 NaN
    
    print(f"合并后的数据范围: {combined_df.index.min()} 到 {combined_df.index.max()}")
    print(f"总交易日数量: {len(combined_df)}")
    
    # 设定开始计算的日期
    start_date = pd.Timestamp('2024-07-08')
    
    # 找到 start_date 在索引中的位置
    # 如果 start_date 不在索引中，找到最近的一个交易日
    if start_date not in combined_df.index:
        print(f"注意: {start_date} 不是交易日，将寻找后续最近的交易日")
        # 找到所有大于等于 start_date 的日期
        future_dates = combined_df.index[combined_df.index >= start_date]
        if len(future_dates) == 0:
            print("错误: 指定开始日期之后没有数据")
            return
        start_date = future_dates[0]
        print(f"实际开始计算日期: {start_date}")
        
    # 获取需要计算的日期列表
    target_dates = combined_df.index[combined_df.index >= start_date]
    print(f"需要计算的交易日数量: {len(target_dates)}")
    
    window_size = 100
    
    for date in target_dates:
        # 获取当前日期的索引位置
        idx = combined_df.index.get_loc(date)
        
        if idx < window_size:
            print(f"日期 {date.date()} 历史数据不足 {window_size} 天，跳过")
            continue
            
        # 获取历史价格数据
        
        # 使用 T-100 到 T-1 的 100 个收益率数据
        # 需要 T-101 到 T-1 的 101 个价格数据
        
        start_idx = idx - window_size - 1
        end_idx = idx # 切片不包含 end_idx，所以是到 idx-1
        
        if start_idx < 0:
             print(f"日期 {date.date()} 历史数据不足 (需要101天价格)，跳过")
             continue
             
        # 获取价格切片
        price_window = combined_df.iloc[start_idx:end_idx]
        
        # 计算收益率
        returns_window = price_window.pct_change().dropna()
        
        if len(returns_window) != window_size:
             print(f"警告: 日期 {date.date()} 收益率样本数量为 {len(returns_window)}，非预期的 {window_size}")
        
        # 计算协方差矩阵
        cov_matrix = returns_window.cov()
        
        # 保存为 npy
        date_str = date.strftime('%Y-%m-%d')
        save_path = os.path.join(save_dir, f'cov_{date_str}.npy')
        np.save(save_path, cov_matrix.values)
        
        # 可选：打印进度
        # print(f"已保存: {save_path}")
        
    print("所有协方差矩阵计算完成。")

if __name__ == "__main__":
    calculate_covariance_matrices()
