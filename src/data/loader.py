import pandas as pd
import os
from src.utils.helpers import load_config
import yfinance as yf
from datetime import datetime, date, timedelta


def download_stock_data(ticker: str, save_path: str):
    """
    Download stock data and save as CSV.
    """
    print(f"📥 Downloading {ticker} data from Yahoo Finance...")
    # df = yf.download(ticker, period=period, interval=interval)

    # Load config
    config = load_config("configs/config.yaml")

    stock = yf.Ticker(
        config["data"]["ticker"]
    )  # initiate instance of Ticker class for the stock
    end_date = datetime(date.today().year, date.today().month, date.today().day)  # today's date
    start_date = end_date - timedelta(365 * 15)  # date a year back from now
    df = stock.history(start=start_date, end=end_date, auto_adjust=False)

    df.reset_index(inplace=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"✅ Saved to {save_path}")
    return df


def load_csv_data(path):
    filepath = os.path.join(path)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at: {filepath}")

    df = pd.read_csv(filepath)
    df = df.dropna()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")

    # Keep relevant features; use "Adj Close" as target
    df = df[["Adj Close", "Open", "High", "Low", "Close", "Volume"]]

    return df
