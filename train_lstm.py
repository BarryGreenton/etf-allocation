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
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def train_and_predict(data_file_path, current_dir):
    etf_code = os.path.basename(data_file_path).replace('.csv', '')
    print(f'\n{'=' * 20} Processing: {etf_code} {'=' * 20}')
    print(f'Reading: {data_file_path}')
    df = pd.read_csv(data_file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.iloc[250:].reset_index(drop=True)
    feature_indices = list(range(1, 18)) + [19]
    feature_cols = df.columns[feature_indices].tolist()
    target_col_idx = 1
    print(f'Feature columns: {feature_cols}')
    print(f'Feature count: {len(feature_cols)}')
    close_price = df.iloc[:, target_col_idx]
    close_diff = close_price.diff()
    close_diff = close_diff.fillna(0)
    test_days = 250
    train_size = len(df) - test_days
    if train_size <= 0:
        print(f'Insufficient test data ({etf_code})')
        return
    train_df = df.iloc[:train_size]
    print(f'Training set: {len(train_df)} days')
    print(f'Test set: {test_days} days')
    scaler_X = StandardScaler()
    train_features = train_df.iloc[:, feature_indices].values
    scaler_X.fit(train_features)
    all_features = df.iloc[:, feature_indices].values
    all_features_scaled = scaler_X.transform(all_features)
    scaler_y = StandardScaler()
    train_diff = close_diff.iloc[:train_size].values.reshape(-1, 1)
    scaler_y.fit(train_diff)
    all_diff_scaled = scaler_y.transform(close_diff.values.reshape(-1, 1))
    diff_mean = scaler_y.mean_[0]
    diff_scale = scaler_y.scale_[0]
    sequence_length = 30

    def create_dataset(features_scaled, diff_scaled, raw_prices, start_idx, end_idx):
        X = []
        y = []
        base_prices = []
        for i in range(start_idx, end_idx):
            if i < sequence_length:
                continue
            X.append(features_scaled[i - sequence_length:i])
            y.append(diff_scaled[i, 0])
            base_prices.append(raw_prices[i - 1])
        return (np.array(X), np.array(y), np.array(base_prices))
    raw_close_prices = df.iloc[:, target_col_idx].values
    print('Preparing training sequences...')
    X_train, y_train, _ = create_dataset(all_features_scaled, all_diff_scaled, raw_close_prices, sequence_length, train_size)
    print('Preparing test sequences...')
    X_test, y_test_scaled, base_prices_test = create_dataset(all_features_scaled, all_diff_scaled, raw_close_prices, train_size, len(df))
    test_dates = df.iloc[train_size:, 0].values
    y_test_real = df.iloc[train_size:, target_col_idx].values
    print(f'X_train shape: {X_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'X_test shape: {X_test.shape}')
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='rmsprop')
    print('Training model...')
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
    model_save_dir = os.path.join(current_dir, 'model_checkpoints')
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    model_path = os.path.join(model_save_dir, f'lstm_{etf_code}_diff.h5')
    model.save(model_path)
    print(f'Saved model: {model_path}')
    print('Predicting test observations...')
    pred_diff_scaled = model.predict(X_test)
    pred_diff_scaled = pred_diff_scaled.flatten()
    pred_diff = pred_diff_scaled * diff_scale + diff_mean
    y_pred = base_prices_test + pred_diff
    mse = mean_squared_error(y_test_real, y_pred)
    print(f'Test close MSE: {mse:.4f}')
    true_returns = (y_test_real - base_prices_test) / base_prices_test
    pred_returns = pred_diff / base_prices_test
    returns_se = (true_returns - pred_returns) ** 2
    returns_mse = np.mean(returns_se)
    print(f'Test return MSE: {returns_mse:.6f}')
    result_save_dir = os.path.join(current_dir, 'test_results')
    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)
    result_df = pd.DataFrame({'date': test_dates, 'actual_close': y_test_real, 'predicted_close': y_pred, 'previous_close': base_prices_test, 'predicted_change': pred_diff, 'actual_return': true_returns, 'predicted_return': pred_returns, 'squared_return_error': returns_se})
    result_csv_path = os.path.join(result_save_dir, f'{etf_code}_test_predictions.csv')
    result_df.to_csv(result_csv_path, index=False, encoding='utf-8-sig')
    print(f'Saved test results: {result_csv_path}')
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, y_test_real, label='actual_close', color='blue')
    plt.plot(test_dates, y_pred, label='predicted_close', color='red', linestyle='--')
    plt.title(f'{etf_code} close price forecasts')
    plt.xlabel('Date')
    plt.ylabel('Close')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plot_path = os.path.join(result_save_dir, f'{etf_code}_prediction_plot.png')
    plt.savefig(plot_path)
    plt.close()
    print(f'Saved close plot: {plot_path}')
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, true_returns, label='actual_return', color='green', alpha=0.7)
    plt.plot(test_dates, pred_returns, label='predicted_return', color='orange', linestyle='--', alpha=0.7)
    plt.title(f'{etf_code} daily return forecasts')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plot_path_returns = os.path.join(result_save_dir, f'{etf_code}_returns_plot.png')
    plt.savefig(plot_path_returns)
    plt.close()
    print(f'Saved return plot: {plot_path_returns}')

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = current_dir
    data_dir = os.path.join(project_root, 'data', 'etf-features')
    if not os.path.exists(data_dir):
        print(f'Data directory missing: {data_dir}')
        return
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        print('No CSV files found')
        return
    print(f'Found {len(csv_files)} files for training')
    for csv_file in csv_files:
        file_path = os.path.join(data_dir, csv_file)
        try:
            train_and_predict(file_path, current_dir)
        except Exception as e:
            print(f'Error processing {csv_file}: {str(e)}')
            import traceback
            traceback.print_exc()
if __name__ == '__main__':
    main()
