import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def scale_data(df, scaler_type="minmax"):
    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError("Unsupported scaler type")

    # Ensure "Adj Close" is first (target), rest are features
    ordered_cols = ["Adj Close"] + [col for col in df.columns if col != "Adj Close"]
    df = df[ordered_cols]

    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler


def create_lstm_windows(data: np.ndarray, window_size: int, horizon: int):
    """
    Create sliding windows for LSTM input and targets.

    Args:
        data (np.ndarray): Scaled time series data.
        window_size (int): Lookback window size.
        horizon (int): Forecast horizon.

    Returns:
        X (np.ndarray): Inputs of shape (samples, window_size, features)
        y (np.ndarray): Targets of shape (samples,)
    """
    X, y = [], []

    for i in range(len(data) - window_size - horizon + 1):
        # fmt: off
        X.append(data[i:i + window_size])
        # fmt: on
        y.append(data[i + window_size + horizon - 1][3])  # Assuming 'Close' is column index 3

    return np.array(X), np.array(y)


def split_data(X, y, split_ratio=0.8):
    """
    Split data into training and testing sets.

    Returns:
        X_train, y_train, X_test, y_test
    """
    split_idx = int(len(X) * split_ratio)
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]
