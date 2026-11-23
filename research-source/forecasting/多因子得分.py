import os
import pandas as pd
import numpy as np

def main():
    # 获取当前脚本所在目录 (收益预测)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 训练用数据文件夹路径 (在数据工程文件夹下)
    project_root = os.path.dirname(current_dir) # .../实验
    data_dir = os.path.join(project_root, '数据工程', '训练用数据')
    
    # 检查文件夹是否存在
    if not os.path.exists(data_dir):
        # 如果相对路径找不到，尝试绝对路径
        absolute_data_dir = "data/训练用数据"
        if os.path.exists(absolute_data_dir):
            data_dir = absolute_data_dir
        else:
            print(f"错误：未找到文件夹 {data_dir} 或 {absolute_data_dir}")
            return
    
    # 获取所有 ETF 数据文件
    etf_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if len(etf_files) != 10:
        print(f"警告：预期 10 个 ETF 文件，实际找到 {len(etf_files)} 个。")
    
    # 读取所有 ETF 数据到字典中，键为 ETF 代码 (文件名去掉 .csv)，值为 DataFrame
    etf_data = {}
    for f in etf_files:
        etf_code = f.replace('.csv', '')
        file_path = os.path.join(data_dir, f)
        df = pd.read_csv(file_path)
        # 确保日期列为 datetime 类型，方便排序和匹配
        df['date'] = pd.to_datetime(df['date'])
        etf_data[etf_code] = df
    
    # 获取所有 ETF 的公共交易日期
    common_dates = etf_data[etf_files[0].replace('.csv', '')]['date']
    for etf_code in etf_data:
        common_dates = np.intersect1d(common_dates, etf_data[etf_code]['date'])
    common_dates = np.sort(common_dates)
    
    print(f"共有 {len(common_dates)} 个公共交易日。")
    
    # 获取示例 DataFrame 以确定列名
    sample_df = etf_data[etf_files[0].replace('.csv', '')]
    print("当前表格列名:", sample_df.columns.tolist())
    
    # 定义因子列名列表 (共 16 个)
    # 列顺序如下：
    # date, close, pct_chg, mtm_5, mtm_10, mtm_20, bias_ma5, bias_ma10, bias_ma20, tr, atr, rs, rsi, bias_ema12, bias_ema26, dif, dea, volume_change_rate, future_1d_return
    
    # 因子列 (索引 2 到 17):
    # 'pct_chg', 'mtm_5', 'mtm_10', 'mtm_20', 'bias_ma5', 'bias_ma10', 'bias_ma20', 'tr', 'atr', 'rs', 'rsi', 'bias_ema12', 'bias_ema26', 'dif', 'dea', 'volume_change_rate'
    
    factor_columns = sample_df.columns[2:18].tolist()
    print("使用的因子列:", factor_columns)
    
    # 未来收益率列名
    future_return_column = 'future_1d_return'
    if future_return_column not in sample_df.columns:
         # 尝试根据索引获取
         if len(sample_df.columns) > 18:
             future_return_column = sample_df.columns[18]
             print(f"警告：未找到列 {future_return_column}，使用第 19 列 {future_return_column} 作为收益率列")
         else:
             print("错误：无法找到未来收益率列")
             return
             
    print(f"使用的收益率列: {future_return_column}")

    # 预处理：将数据按日期对齐
    sorted_etf_codes = sorted(etf_data.keys())
    num_etfs = len(sorted_etf_codes)
    num_days = len(common_dates)
    num_factors = len(factor_columns)
    
    factors_tensor = np.zeros((num_days, num_etfs, num_factors))
    returns_matrix = np.zeros((num_days, num_etfs))
    
    for i, etf_code in enumerate(sorted_etf_codes):
        df = etf_data[etf_code]
        # set_index 后列会变化，所以要在 reindex 之后根据列名取值
        df_indexed = df.set_index('date').reindex(common_dates)
        
        # 使用列名提取数据，避免索引混淆
        factors_tensor[:, i, :] = df_indexed[factor_columns].values
        returns_matrix[:, i] = df_indexed[future_return_column].values

    final_scores = np.zeros((num_days, num_etfs))
    window_size = 250
    update_interval = 20
    
    print("开始计算多因子模型权重和得分...")
    current_weights = np.zeros(num_factors)
    
    for t in range(window_size, num_days):
        if (t - window_size) % update_interval == 0:
            start_idx = t - window_size
            end_idx = t
            factor_ics = np.zeros((window_size, num_factors))
            
            for k in range(num_factors):
                for d_offset in range(window_size):
                    d_idx = start_idx + d_offset
                    f_values = factors_tensor[d_idx, :, k]
                    r_values = returns_matrix[d_idx, :]
                    
                    valid_mask = ~np.isnan(f_values) & ~np.isnan(r_values)
                    if np.sum(valid_mask) < 2:
                        factor_ics[d_offset, k] = 0
                        continue
                        
                    f_valid = f_values[valid_mask]
                    r_valid = r_values[valid_mask]
                    
                    if np.std(f_valid) == 0:
                        f_z = np.zeros_like(f_valid)
                    else:
                        f_z = (f_valid - np.mean(f_valid)) / np.std(f_valid)
                    
                    ic = np.corrcoef(f_z, r_valid)[0, 1]
                    if np.isnan(ic): ic = 0
                    factor_ics[d_offset, k] = ic
            
            ic_means = np.mean(factor_ics, axis=0)
            ic_stds = np.std(factor_ics, axis=0)
            ir_values = np.divide(ic_means, ic_stds, out=np.zeros_like(ic_means), where=ic_stds!=0)
            ir_abs_sum = np.sum(np.abs(ir_values))
            
            if ir_abs_sum == 0:
                current_weights = np.zeros(num_factors)
            else:
                current_weights = ir_values / ir_abs_sum
                
        day_factor_scores = np.zeros((num_etfs, num_factors))
        for k in range(num_factors):
            f_values = factors_tensor[t, :, k]
            if np.std(f_values) == 0:
                f_z = np.zeros_like(f_values)
            else:
                f_z = (f_values - np.mean(f_values)) / np.std(f_values)
            day_factor_scores[:, k] = f_z
            
        final_scores[t, :] = np.dot(day_factor_scores, current_weights)
        
    print("计算完成，正在保存结果...")
    
    for i, etf_code in enumerate(sorted_etf_codes):
        score_series = pd.Series(final_scores[:, i], index=common_dates, name='multi_factor_score')
        score_series.iloc[:window_size] = np.nan
        
        file_path = os.path.join(data_dir, f"{etf_code}.csv")
        original_df = pd.read_csv(file_path)
        original_df['date'] = pd.to_datetime(original_df['date'])
        
        if 'multi_factor_score' in original_df.columns:
            original_df = original_df.drop(columns=['multi_factor_score'])
        
        merged_df = pd.merge(original_df, score_series, left_on='date', right_index=True, how='left')
        merged_df.to_csv(file_path, index=False)
        print(f"已更新文件: {file_path}")
        
    print(f"所有文件已更新完毕，存储在 {data_dir}")

if __name__ == "__main__":
    main()
