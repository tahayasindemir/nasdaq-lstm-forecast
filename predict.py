import torch
import numpy as np
import pandas as pd
from src.utils.helpers import load_config
from src.data.loader import load_csv_data
from src.data.preprocessing import scale_data
from src.models.lstm import LSTMModel
import matplotlib.pyplot as plt
import os


def predict_future(model, initial_window, steps, device):
    """
    Predict future time steps recursively.

    Args:
        model: Trained LSTM model.
        initial_window (np.ndarray): Shape (window_size, num_features)
        steps (int): Number of future days to forecast.
        device (str): 'cpu' or 'cuda'.

    Returns:
        np.ndarray: Forecasted values (steps, 1)
    """
    model.eval()
    input_seq = initial_window.copy()
    predictions = []

    for _ in range(steps):
        input_tensor = torch.tensor(input_seq[np.newaxis, :, :], dtype=torch.float32).to(device)
        with torch.no_grad():
            pred = model(input_tensor)
        pred_value = pred.cpu().numpy().flatten()[0]
        predictions.append(pred_value)

        # Append prediction and slide window
        next_input = np.append(input_seq[1:], [[pred_value] * input_seq.shape[1]], axis=0)
        input_seq = next_input

    return np.array(predictions)


def main():
    config = load_config("configs/config.yaml")
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")

    # Load and scale full data
    df = load_csv_data(config["data"]["data_path"])
    scaled_data, scaler = scale_data(df, config["data"]["scaler"])

    # Prepare the last window from scaled data
    window_size = config["data"]["window_size"]
    input_window = scaled_data[-window_size:]  # shape: (window_size, num_features)

    # Model
    model_cfg = config["model"]
    model = LSTMModel(
        input_size=model_cfg["input_size"],
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
    ).to(device)

    model_path = f"models/{config['data']['ticker']}_lstm.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Loaded model from {model_path}")

    # Predict 252 days
    forecast_scaled = predict_future(model, input_window, steps=252, device=device)

    # Inverse scale predictions
    dummy_input = np.zeros((252, scaled_data.shape[1]))
    dummy_input[:, 0] = forecast_scaled  # Assume forecasting "Close"
    forecast_inverse = scaler.inverse_transform(dummy_input)[:, 0]

    # Create future date index
    last_date = df.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=252)

    forecast_df = pd.DataFrame(
        {"Date": future_dates, "Forecast_Close": forecast_inverse}
    ).set_index("Date")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[-100:], df["Close"].values[-100:], label="Recent Close")
    plt.plot(forecast_df.index, forecast_df["Forecast_Close"], label="Forecast (1 Year Ahead)")
    plt.title(f"{config['data']['ticker']} LSTM Forecast - 1 Year Ahead")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{config['data']['ticker']}_forecast.png")
    plt.show()

    print(f"✅ Forecast saved to plots/{config['data']['ticker']}_forecast.png")


if __name__ == "__main__":
    main()
