import pandas as pd
import numpy as np
import os

def load_and_prepare_data(stock_filename, fed_funds_filename):
    """Load and prepare the data for training the agent."""
    # Load stock data
    df = pd.read_csv(stock_filename, index_col='Date', parse_dates=True)

    # Extract time-based features
    df['Weekday'] = df.index.weekday
    df['Month'] = df.index.month
    df['Quarter'] = df.index.quarter
    df['Day_Sine'] = np.sin(2 * np.pi * df.index.dayofyear / 365.0)
    df['Day_Cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.0)

    # month sine and cosine features
    df['Month_Sine'] = np.sin(2 * np.pi * df.index.month / 12.0)
    df['Month_Cos'] = np.cos(2 * np.pi * df.index.month / 12.0)

    # Load federal funds rate data
    fed_funds_df = pd.read_csv(fed_funds_filename, index_col='Date', parse_dates=True)

    # Add savings rate to the dataframe and name it 'Savings_Rate'
    df = df.join(fed_funds_df, how='left')
    df.rename(columns={'Value': 'Savings_Rate'}, inplace=True)

    # Fill missing values, using obj.ffill()
    df.ffill(inplace=True)  # Forward fill missing federal funds rate values

    # Ensure no missing values remain
    assert df.isnull().sum().sum() == 0, "There are still missing values in the dataframe."

    return df

def split_data(data, train_ratio=0.8):
    """Split the data into training and testing datasets."""
    train_size = int(len(data) * train_ratio)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    return train_data, test_data

if __name__ == "__main__":
    df = load_and_prepare_data(os.path.join("data", "AAPL.csv"), os.path.join("data", "fed_funds_rate.csv"))
    train_df, test_df = split_data(df)

    # Save the training and testing datasets to separate CSV files
    train_df.to_csv(os.path.join("data", "processed_train_data.csv"))
    test_df.to_csv(os.path.join("data", "processed_test_data.csv"))