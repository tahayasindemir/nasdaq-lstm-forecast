import torch
from src.utils.helpers import load_config
from src.data.loader import download_stock_data
from src.data.preprocessing import scale_data, create_lstm_windows, split_data
from src.models.lstm import LSTMModel
from src.training.trainer import prepare_dataloader, train_one_epoch, evaluate
import os


def main():
    # Load config
    config = load_config("configs/config.yaml")
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")

    # Load or fetch data
    df = download_stock_data(config["data"]["ticker"], config["data"]["data_path"])
    # df = load_csv_data(config["data"]["data_path"], ticker=config["data"]["ticker"])

    # Preprocess data
    df = df.drop(columns=["Date", "Dividends", "Stock Splits"])  # drop non-numeric
    # Put target 'Adj Close' first
    target = "Adj Close"
    df = df[[target] + [col for col in df.columns if col != target]]

    scaled_data, scaler = scale_data(df, config["data"]["scaler"])
    X, y = create_lstm_windows(
        scaled_data, config["data"]["window_size"], config["data"]["horizon"]
    )
    X_train, y_train, X_test, y_test = split_data(X, y, config["data"]["split_ratio"])

    # DataLoaders
    train_loader = prepare_dataloader(
        X_train, y_train, config["training"]["batch_size"], device
    )
    test_loader = prepare_dataloader(X_test, y_test, config["training"]["batch_size"], device)

    # Model
    model_cfg = config["model"]
    model = LSTMModel(
        input_size=model_cfg["input_size"],
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
    ).to(device)

    # Loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    # Training loop
    for epoch in range(config["training"]["epochs"]):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        test_loss = evaluate(model, test_loader, criterion)

        print(f"Epoch {epoch+1:02} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/{config['data']['ticker']}_lstm.pth")
    print(f"✅ Model saved to models/{config['data']['ticker']}_lstm.pth")


if __name__ == "__main__":
    main()
