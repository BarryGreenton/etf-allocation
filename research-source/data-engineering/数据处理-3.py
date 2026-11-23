import os
import pandas as pd
import numpy as np

def main():
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定义训练用数据文件夹路径
    target_dir = os.path.join(current_dir, '训练用数据')
    
    # 检查文件夹是否存在
    if not os.path.exists(target_dir):
        print(f"错误：未找到文件夹 {target_dir}")
        return
        
    # 指定只处理这个文件
    target_file = '518880.SH.csv'
    data_files = [target_file]
    
    print(f"开始处理训练用数据，共 {len(data_files)} 个文件...")
    
    for file_name in data_files:
        file_path = os.path.join(target_dir, file_name)
        
        if not os.path.exists(file_path):
            print(f"错误：未找到文件 {file_path}")
            continue

        try:
            # 读取 CSV 文件
            df = pd.read_csv(file_path)
            
            # 确认是否有足够的列
            # 原始列索引从 0 开始
            # 第 5 列 (index 4): close
            # 第 14 列 (index 13): ma_5
            # 第 15 列 (index 14): ma_10
            # 第 16 列 (index 15): ma_20
            # 第 28 列 (index 27): ema_12
            # 第 29 列 (index 28): ema_26
            
            # 定义需要计算 Bias 的列及其对应的目标名称
            # 格式: (原列索引, 新列名)
            bias_config = [
                (13, 'bias_ma5'),
                (14, 'bias_ma10'),
                (15, 'bias_ma20'),
                (27, 'bias_ema12'),
                (28, 'bias_ema26')
            ]
            
            close_col_idx = 4
            
            # 检查列数是否足够
            max_idx = max([x[0] for x in bias_config] + [close_col_idx])
            if df.shape[1] <= max_idx:
                print(f"警告：文件 {file_name} 列数不足 (需要至少 {max_idx + 1} 列，实际 {df.shape[1]} 列)，跳过处理")
                continue
            
            # 获取收盘价列
            close_prices = df.iloc[:, close_col_idx].copy() # Copy to avoid SettingWithCopy warning/issues if df slice
            
            # 计算 Bias 并替换原列
            # 为了避免修改列名时影响后续索引访问（虽然这里是按位置访问，但修改 df.columns 可能会有副作用），
            # 我们先计算所有新值，最后再一次性修改列名或替换列
            
            new_values = {}
            
            for col_idx, new_name in bias_config:
                ma_values = df.iloc[:, col_idx]
                # 计算 bias = (close - ma) / ma
                # 注意处理除以零的情况 (虽然均线一般不为0)
                bias_values = (close_prices - ma_values) / ma_values
                
                # 存储新值和列名
                new_values[col_idx] = (bias_values, new_name)
            
            # 应用新值和新列名
            # 需要注意的是，pandas 的列名是 Index 对象，不能直接修改单个元素
            new_columns = list(df.columns)
            for col_idx, (val, name) in new_values.items():
                df.iloc[:, col_idx] = val
                new_columns[col_idx] = name
            
            df.columns = new_columns
                
            # 定义需要删除的列索引 (从 1 开始计数转换为 0 开始的索引)
            # 第 2, 3, 4, 6, 7, 9, 10, 17, 18, 19, 22, 23, 26, 27, 31 列
            # 对应索引: 1, 2, 3, 5, 6, 8, 9, 16, 17, 18, 21, 22, 25, 26, 30
            cols_to_remove_indices = [1, 2, 3, 5, 6, 8, 9, 16, 17, 18, 21, 22, 25, 26, 30]
            
            # 确保索引在有效范围内
            valid_remove_indices = [i for i in cols_to_remove_indices if i < df.shape[1]]
            
            # 获取列名以便删除
            # 使用 iloc 选择要保留的列更安全，因为 drop by name 可能会遇到重复列名的问题（虽然这里应该没有）
            # 或者直接获取要删除的列名列表
            cols_to_remove = df.columns[valid_remove_indices]
            
            # 删除列
            df.drop(columns=cols_to_remove, inplace=True)
            
            # 保存回文件
            df.to_csv(file_path, index=False)
            print(f"已处理: {file_name}")
            
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {e}")
            
    print("所有文件处理完成。")

if __name__ == "__main__":
    main()
