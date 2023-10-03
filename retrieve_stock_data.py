import yfinance as yf
import pandas as pd
import os


def get_sp500_tickers():
    """Get S&P 500 tickers from Wikipedia and return a list of tickers."""
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = table[0]
    tickers = df['Symbol'].tolist()
    return tickers


def download_data(ticker, start_date='2000-01-01', end_date='2023-01-01'):
    """Download stock data for the given ticker and time period from Yahoo Finance."""
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data


def save_data_to_csv(data, ticker):
    """Save the stock data to a CSV file."""
    if not os.path.exists('data'):
        os.mkdir('data')
    filename = f"data/{ticker}.csv"
    data.to_csv(filename)


if __name__ == "__main__":
    tickers = get_sp500_tickers()

    for ticker in tickers:
        print(f"Downloading data for {ticker}...")
        data = download_data(ticker)
        save_data_to_csv(data, ticker)
        print(f"Saved data for {ticker} to data/{ticker}.csv")