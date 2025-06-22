# NASDAQ Stock Price Forecasting with LSTM

📈 A deep learning-based project to forecast NASDAQ stock prices using an LSTM (Long Short-Term Memory) model. Designed with modularity and scalability in mind — easily extendable with other models like XGBoost or Transformer-based architectures.

---

## 🚀 Project Overview

This project uses historical stock data to train an LSTM neural network that predicts future prices based on past patterns. The pipeline includes:

- Data collection (from Yahoo Finance)
- Preprocessing and feature scaling
- Sequence generation for LSTM input
- Model training and saving
- Forecasting and evaluation (including MAE, RMSE, MAPE, directional accuracy)

---

## 🧱 Folder Structure

nasdaq-lstm-forecast/
├── configs/ # YAML config files
├── data/ # Raw & processed stock data
├── models/ # Trained LSTM model checkpoints
├── notebooks/ # EDA or experiment notebooks
├── plots/ # Forecast and evaluation charts
├── src/ # All source code
│ ├── data/ # Data loading & preprocessing
│ ├── models/ # LSTM model definitions
│ └── utils/ # Configs, helpers
├── train.py # Model training script
├── predict.py # 1-year ahead forecasting
├── evaluate.py # Metrics + evaluation plot
├── requirements.txt # Python dependencies
└── README.md # This file


---

## 🔧 Setup & Installation

1. **Clone the repo**:
   ```bash
   git clone https://github.com/tahayasindemir/nasdaq-lstm-forecast.git
   cd nasdaq-lstm-forecast


📈 Extending This Project

Swap LSTM for XGBoost, GRU, or Transformers
Add technical indicators (RSI, MACD, etc.)
Use log returns instead of raw prices
Convert into a Streamlit app for interactive demos


🙋‍♂️ Contributions Welcome

Feel free to fork, open issues, or submit PRs if you’d like to improve or extend this project.
