
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error

# 设置 matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def process_results():
    # 路径设置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, 'test_results')
    
    # 创建保存高清图表的文件夹
    plots_dir = os.path.join(results_dir, 'high_res_plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        
    # 汇总结果的文件路径
    summary_file_path = os.path.join(results_dir, 'summary_metrics.txt')
    
    # 获取所有结果 CSV 文件
    csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("未找到结果文件")
        return

    print(f"找到 {len(csv_files)} 个结果文件，开始处理...")
    
    with open(summary_file_path, 'w', encoding='utf-8') as summary_file:
        summary_file.write("文件名\t收盘价预测MSE\t日收益率预测MSE\n")
        
        for csv_file in csv_files:
            file_path = os.path.join(results_dir, csv_file)
            etf_code = csv_file.replace('_test_predictions.csv', '')
            
            print(f"正在处理: {etf_code}")
            
            # 读取 CSV
            try:
                # 尝试用 utf-8-sig 读取 (处理 BOM)
                df = pd.read_csv(file_path, encoding='utf-8-sig')
            except UnicodeDecodeError:
                try:
                    # 尝试用 gbk 读取
                    df = pd.read_csv(file_path, encoding='gbk')
                except Exception as e:
                    print(f"  读取文件 {csv_file} 失败: {str(e)}")
                    continue
            except Exception as e:
                print(f"  读取文件 {csv_file} 失败: {str(e)}")
                continue
            
            # 确保日期列格式正确 (第1列，索引0)
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            dates = df.iloc[:, 0]
            
            # 获取数据列 (根据用户描述和之前的代码逻辑)
            # 第2列 (index 1): 真实收盘价
            real_close = df.iloc[:, 1]
            # 第3列 (index 2): 预测收盘价
            pred_close = df.iloc[:, 2]
            # 第6列 (index 5): 真实日收益率
            real_return = df.iloc[:, 5]
            # 第7列 (index 6): 预测日收益率
            pred_return = df.iloc[:, 6]
            
            # --- 1. 计算 MSE ---
            mse_close = mean_squared_error(real_close, pred_close)
            mse_return = mean_squared_error(real_return, pred_return)
            
            # 写入汇总文件
            summary_file.write(f"{etf_code}\t{mse_close:.6f}\t{mse_return:.6f}\n")
            
            # --- 2. 绘制高清对比图 ---
            
            # 图1: 收盘价对比
            plt.figure(figsize=(15, 8), dpi=300) # 提高分辨率和尺寸
            plt.plot(dates, real_close, label='真实收盘价', color='blue', linewidth=1.5)
            plt.plot(dates, pred_close, label='预测收盘价', color='red', linestyle='--', linewidth=1.5)
            
            plt.title(f'{etf_code} 收盘价预测对比', fontsize=16)
            plt.xlabel('日期', fontsize=12)
            plt.ylabel('收盘价', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, which='both', linestyle='--', alpha=0.7)
            
            # 调整纵坐标范围，使线条差异更明显
            # 使用自动缩放，但留一点余地
            plt.margins(y=0.1)
            
            # 格式化日期轴
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gcf().autofmt_xdate()
            
            plot_path_close = os.path.join(plots_dir, f'{etf_code}_close_price_high_res.png')
            plt.savefig(plot_path_close, bbox_inches='tight')
            plt.close()
            
            # 图2: 收益率对比
            plt.figure(figsize=(15, 8), dpi=300)
            plt.plot(dates, real_return, label='真实日收益率', color='green', alpha=0.8, linewidth=1.5)
            plt.plot(dates, pred_return, label='预测日收益率', color='orange', linestyle='--', alpha=0.8, linewidth=1.5)
            
            plt.title(f'{etf_code} 日收益率预测对比', fontsize=16)
            plt.xlabel('日期', fontsize=12)
            plt.ylabel('收益率', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, which='both', linestyle='--', alpha=0.7)
            
            # 调整纵坐标范围
            plt.margins(y=0.1)
            
            # 格式化日期轴
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gcf().autofmt_xdate()
            
            plot_path_return = os.path.join(plots_dir, f'{etf_code}_return_high_res.png')
            plt.savefig(plot_path_return, bbox_inches='tight')
            plt.close()
            
            # --- 3. 删除第8列 (index 7) ---
            # 检查是否有第8列
            if df.shape[1] >= 8:
                # 删除 index 7 的列
                df_dropped = df.drop(df.columns[7], axis=1)
                # 保存回原文件
                df_dropped.to_csv(file_path, index=False, encoding='utf-8-sig')
                print(f"  已更新文件 (删除第8列): {csv_file}")
            else:
                print(f"  警告: 文件 {csv_file} 列数不足8列，未执行删除操作")
                
    print(f"\n处理完成！汇总结果已保存至: {summary_file_path}")
    print(f"高清图表已保存至: {plots_dir}")

if __name__ == "__main__":
    process_results()
