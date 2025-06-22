import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

from src.utils.helpers import load_config
from src.data.loader import load_csv_data
from src.data.preprocessing import scale_data
from src.models.lstm import LSTMModel


def predict_sequence(model, input_seq, steps, device):
    model.eval()
    input_seq = input_seq.copy()
    predictions = []

    for _ in range(steps):
        input_tensor = torch.tensor(input_seq[np.newaxis, :, :], dtype=torch.float32).to(device)
        with torch.no_grad():
            pred = model(input_tensor)
        pred_val = pred.cpu().numpy().flatten()[0]
        predictions.append(pred_val)

        # slide window
        input_seq = np.append(input_seq[1:], [[pred_val] * input_seq.shape[1]], axis=0)

    return np.array(predictions)


def main():
    config = load_config("configs/config.yaml")
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")

    df = load_csv_data(config["data"]["data_path"])
    scaled_data, scaler = scale_data(df, config["data"]["scaler"])

    window_size = config["data"]["window_size"]
    test_steps = 60  # how far ahead to forecast for evaluation

    # Split data
    train_data = scaled_data[:-test_steps]
    actual_prices = df["Close"].values[-test_steps:]

    # Prepare model
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

    # Prepare input window from last part of training data
    input_window = train_data[-window_size:]

    # Predict
    pred_scaled = predict_sequence(model, input_window, steps=test_steps, device=device)

    # Inverse scale predictions
    dummy_input = np.zeros((test_steps, scaled_data.shape[1]))
    dummy_input[:, 0] = pred_scaled  # assuming "Close" is first
    pred_prices = scaler.inverse_transform(dummy_input)[:, 0]

    # Metrics
    mae = mean_absolute_error(actual_prices, pred_prices)
    rmse = np.sqrt(mean_squared_error(actual_prices, pred_prices))
    mape = mean_absolute_percentage_error(actual_prices, pred_prices)

    correct_direction = (np.sign(np.diff(actual_prices)) == np.sign(np.diff(pred_prices))).sum()
    directional_acc = correct_direction / (len(actual_prices) - 1)

    print("\n📊 Evaluation Metrics:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4%}")
    print(f"Directional Accuracy: {directional_acc:.2%}")

    # Plot
    dates = df.index[-test_steps:]
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual_prices, label="Actual")
    plt.plot(dates, pred_prices, label="Predicted")
    plt.title(f"{config['data']['ticker']} Forecast Evaluation (Last {test_steps} Days)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/{config['data']['ticker']}_evaluation.png")
    plt.show()
    print(f"✅ Evaluation plot saved to plots/{config['data']['ticker']}_evaluation.png")


if __name__ == "__main__":
    main()
