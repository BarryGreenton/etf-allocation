import os
import pandas as pd
import numpy as np

def main():
    # Define paths
    # Assuming script is in "数据工程" and data is in "数据工程/数据"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '数据')

    # Ensure data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        return

    # Get list of CSV files
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    print(f"Found {len(files)} CSV files. Starting processing...")

    total_files = 0
    
    for filename in files:
        file_path = os.path.join(data_dir, filename)
        
        try:
            # Read CSV
            # Using default encoding. If fails, might need 'gbk'
            df = pd.read_csv(file_path)
            
            # 1. Convert data types to numeric
            # Apply to_numeric to all columns, coercing errors to NaN
            df = df.apply(pd.to_numeric, errors='coerce')
            
            # 2. Calculate future 1-day return
            # Based on 4th column (index 3). 
            # Formula: (Close_{t+1} - Close_t) / Close_t
            # shift(-1) brings next row's value to current row
            if df.shape[1] >= 4:
                close_col_name = df.columns[3]
                # Ensure we are using the correct column even if names change, but iloc is safer for "4th column"
                close_prices = df.iloc[:, 3]
                
                # Calculate return
                # Note: The last row will be NaN because there is no future data
                df['future_1d_return'] = close_prices.shift(-1) / close_prices - 1
            else:
                print(f"Warning: {filename} has fewer than 4 columns.")
            
            # 3. Remove 21st and 22nd columns
            # Indices 20 and 21 (0-based)
            # We identify them by index. Note: Adding a column appends to the end, 
            # so original indices 20 and 21 are preserved until we drop.
            if df.shape[1] >= 22:
                # Identify columns to drop by position
                cols_to_drop = df.columns[[20, 21]]
                df.drop(columns=cols_to_drop, inplace=True)
            
            # 4. Statistics: Count trading days
            num_days = len(df)
            print(f"Processed {filename}: {num_days} trading days")
            
            # Save processed data back to file
            df.to_csv(file_path, index=False)
            total_files += 1
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"Processing complete. Processed {total_files} files.")

if __name__ == "__main__":
    main()
