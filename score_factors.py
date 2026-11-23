import os
import pandas as pd
import numpy as np

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = current_dir
    data_dir = os.path.join(project_root, 'data', 'etf-features')
    if not os.path.exists(data_dir):
        absolute_data_dir = 'data/etf-features'
        if os.path.exists(absolute_data_dir):
            data_dir = absolute_data_dir
        else:
            print(f'Directory missing: {data_dir} or {absolute_data_dir}')
            return
    etf_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if len(etf_files) != 10:
        print(f'Expected ten ETF files, found {len(etf_files)} files.')
    etf_data = {}
    for f in etf_files:
        etf_code = f.replace('.csv', '')
        file_path = os.path.join(data_dir, f)
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        etf_data[etf_code] = df
    common_dates = etf_data[etf_files[0].replace('.csv', '')]['date']
    for etf_code in etf_data:
        common_dates = np.intersect1d(common_dates, etf_data[etf_code]['date'])
    common_dates = np.sort(common_dates)
    print(f'Found {len(common_dates)} common trading days.')
    sample_df = etf_data[etf_files[0].replace('.csv', '')]
    print('Columns:', sample_df.columns.tolist())
    factor_columns = sample_df.columns[2:18].tolist()
    print('Factor columns:', factor_columns)
    future_return_column = 'future_1d_return'
    if future_return_column not in sample_df.columns:
        if len(sample_df.columns) > 18:
            future_return_column = sample_df.columns[18]
            print(f'Missing column {future_return_column}, using column 19 {future_return_column} for returns')
        else:
            print('Future return column missing')
            return
    print(f'Return column: {future_return_column}')
    sorted_etf_codes = sorted(etf_data.keys())
    num_etfs = len(sorted_etf_codes)
    num_days = len(common_dates)
    num_factors = len(factor_columns)
    factors_tensor = np.zeros((num_days, num_etfs, num_factors))
    returns_matrix = np.zeros((num_days, num_etfs))
    for i, etf_code in enumerate(sorted_etf_codes):
        df = etf_data[etf_code]
        df_indexed = df.set_index('date').reindex(common_dates)
        factors_tensor[:, i, :] = df_indexed[factor_columns].values
        returns_matrix[:, i] = df_indexed[future_return_column].values
    final_scores = np.zeros((num_days, num_etfs))
    window_size = 250
    update_interval = 20
    print('Calculating factor weights and scores...')
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
                    if np.isnan(ic):
                        ic = 0
                    factor_ics[d_offset, k] = ic
            ic_means = np.mean(factor_ics, axis=0)
            ic_stds = np.std(factor_ics, axis=0)
            ir_values = np.divide(ic_means, ic_stds, out=np.zeros_like(ic_means), where=ic_stds != 0)
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
    print('Saving results...')
    for i, etf_code in enumerate(sorted_etf_codes):
        score_series = pd.Series(final_scores[:, i], index=common_dates, name='multi_factor_score')
        score_series.iloc[:window_size] = np.nan
        file_path = os.path.join(data_dir, f'{etf_code}.csv')
        original_df = pd.read_csv(file_path)
        original_df['date'] = pd.to_datetime(original_df['date'])
        if 'multi_factor_score' in original_df.columns:
            original_df = original_df.drop(columns=['multi_factor_score'])
        merged_df = pd.merge(original_df, score_series, left_on='date', right_index=True, how='left')
        merged_df.to_csv(file_path, index=False)
        print(f'Updated file: {file_path}')
    print(f'Updated files in {data_dir}')
if __name__ == '__main__':
    main()
