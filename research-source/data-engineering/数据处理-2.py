import os
import pandas as pd
import random
import shutil

def main():
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定义文件路径
    etf_data_path = os.path.join(current_dir, 'etf_data.csv')
    data_dir = os.path.join(current_dir, '数据')
    target_dir = os.path.join(current_dir, '训练用数据')
    
    # 检查 etf_data.csv 是否存在
    if not os.path.exists(etf_data_path):
        print(f"错误：未找到文件 {etf_data_path}")
        return

    # 读取 etf_data.csv
    try:
        etf_df = pd.read_csv(etf_data_path)
    except Exception as e:
        print(f"读取 etf_data.csv 失败: {e}")
        return

    # 确保代码列为字符串格式，方便后续匹配
    if '代码' not in etf_df.columns or '日期' not in etf_df.columns:
        print("错误：etf_data.csv 缺少 '代码' 或 '日期' 列")
        return
        
    etf_df['代码'] = etf_df['代码'].astype(str)
    
    # 获取数据文件夹中的所有 csv 文件
    if not os.path.exists(data_dir):
        print(f"错误：未找到数据文件夹 {data_dir}")
        return
        
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    max_days = 0
    max_days_etf_info = []  # 使用列表存储 (代码, 文件名) 元组
    
    print(f"开始处理，共发现 {len(data_files)} 个数据文件...")
    
    for file_name in data_files:
        file_path = os.path.join(data_dir, file_name)
        
        # 从文件名中提取 ETF 代码 (例如 "159001.SZ.csv" -> "159001")
        etf_code = file_name.split('.')[0]
        
        # 在 etf_data 中查找对应代码的所有行
        etf_dates = etf_df[etf_df['代码'] == etf_code]['日期'].values
        
        if len(etf_dates) == 0:
            print(f"警告：在 etf_data.csv 中未找到代码 {etf_code} 的日期数据")
            continue
            
        try:
            # 读取单个 ETF 数据文件
            df_file = pd.read_csv(file_path)
            
            # 检查行数是否匹配
            if len(df_file) != len(etf_dates):
                print(f"行数不匹配: ETF代码 {etf_code} (数据文件行数: {len(df_file)}, 日期数据行数: {len(etf_dates)}) - 跳过处理")
                continue
                
            # 将日期作为第一列插入
            # 如果 'date' 列已存在，先删除或直接覆盖
            if 'date' in df_file.columns:
                df_file = df_file.drop(columns=['date'])
            
            df_file.insert(0, 'date', etf_dates)
            
            # 统计交易日数量
            days_count = len(df_file)
            
            # 更新最大交易日数量记录
            if days_count > max_days:
                max_days = days_count
                max_days_etf_info = [(etf_code, file_name)]  # 发现新的最大值，重置列表
            elif days_count == max_days:
                max_days_etf_info.append((etf_code, file_name))  # 发现相同最大值，添加到列表
            
            # 保存回文件
            df_file.to_csv(file_path, index=False)
            
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {e}")
            
    # 输出最终结果
    print("-" * 30)
    print("处理完成。")
    if max_days_etf_info:
        # 提取代码列表用于显示
        max_days_etf_codes = [info[0] for info in max_days_etf_info]
        print(f"交易日数量最多的 ETF 代码有 {len(max_days_etf_codes)} 个:")
        print(f"交易日数量为: {max_days}")
        print("代码列表:", ", ".join(max_days_etf_codes))
        
        # 创建训练用数据文件夹
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            print(f"已创建文件夹: {target_dir}")
        else:
            print(f"文件夹已存在: {target_dir}")
            
        # 随机选择 10 个（如果不足 10 个则全部选择）
        num_to_select = min(10, len(max_days_etf_info))
        selected_info = random.sample(max_days_etf_info, num_to_select)
        
        print(f"正在随机选择 {num_to_select} 个文件复制到训练用数据文件夹...")
        
        for etf_code, file_name in selected_info:
            src_file = os.path.join(data_dir, file_name)
            dst_file = os.path.join(target_dir, file_name)
            try:
                shutil.copy2(src_file, dst_file)
                print(f"已复制: {file_name}")
            except Exception as e:
                print(f"复制文件 {file_name} 失败: {e}")
                
        print("所有文件复制完成。")
        
    else:
        print("未找到任何有效的 ETF 数据。")

if __name__ == "__main__":
    main()
