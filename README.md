# 📈 Tesla Stock Price Forecasting — End-to-End LSTM Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-56d364?style=for-the-badge)

**A production-grade deep learning pipeline for time-series stock price forecasting using LSTM, Bidirectional LSTM, and GRU architectures — with a fully interactive Streamlit dashboard.**

[Features](#-features) · [Results](#-results) · [Quick Start](#-quick-start) · [Project Structure](#-project-structure) · [Architecture](#-model-architectures) · [Deployment](#-streamlit-deployment)

</div>

---

## 🎯 Project Overview

This project implements a complete end-to-end machine learning pipeline for forecasting Tesla (TSLA) stock closing prices using deep learning. The dataset spans **2010–2025** and includes OHLCV data sourced from [Kaggle](https://www.kaggle.com/datasets/iamtanmayshukla/tesla-stocks-dataset).

The pipeline covers everything from raw data ingestion to an interactive web deployment — ready for a GitHub portfolio or production use.

---

## ✨ Features

- **Full ML Pipeline** — data loading, cleaning, EDA, feature engineering, preprocessing, training, evaluation
- **3 Deep Learning Models** — Stacked LSTM, Bidirectional LSTM, GRU with side-by-side comparison
- **Rich Feature Engineering** — MA10, MA50, EMA20, RSI14, 5 lag features, daily return
- **Proper Time-Series Handling** — chronological 80/20 split, no data leakage, sliding window sequences
- **Interactive Streamlit App** — upload CSV, train models live, visualize predictions, download results
- **Production-Ready Code** — modular functions, type hints, docstrings, session state management
- **Plotly Visualizations** — interactive charts for price history, RSI, correlation, predictions, loss curves

---

## 📊 Results

| Model | RMSE ↓ | MAE ↓ |
|---|---|---|
| 🥇 **GRU** | **$14.44** | **$10.86** |
| 🥈 Stacked LSTM | $17.22 | $12.65 |
| 🥉 Bidirectional LSTM | $34.76 | $30.02 |

> **GRU outperforms LSTM variants** on this dataset. Its simpler gating mechanism (reset + update gates vs LSTM's 3 gates) acts as implicit regularisation on noisy financial data, reducing overfitting. The Bidirectional LSTM underperformed due to subtle look-ahead leakage within the sliding window.

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/tesla-lstm-forecasting.git
cd tesla-lstm-forecasting
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get the dataset

Download the Tesla stock CSV from [Kaggle](https://www.kaggle.com/datasets/iamtanmayshukla/tesla-stocks-dataset) and place it in the project root.

### 4a. Run the Streamlit app (recommended)

```bash
streamlit run app.py
```

Then open `http://localhost:8501`, upload your CSV via the sidebar, and click **Train Selected Models**.

### 4b. Run the standalone script

```bash
python tesla_lstm_forecast.py your_file.csv
```

### 4c. Run the notebook

```bash
jupyter notebook tesla_lstm_forecast.ipynb
```

---

## 📁 Project Structure

```
tesla-lstm-forecasting/
│
├── app.py                      # Streamlit web application
├── tesla_lstm_forecast.py      # Standalone pipeline script
├── tesla_lstm_forecast.ipynb   # Jupyter notebook version
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── models/                     # Saved model files (after training)
│   ├── model_gru.keras
│   ├── model_lstm.keras
│   └── model_bilstm.keras
│
└── outputs/                    # Generated plots
    ├── eda_overview.png
    ├── predictions.png
    └── loss_curves.png
```

---

## 🧠 Model Architectures

All three models share the same Dense head and are compiled with `Adam` optimizer and `MSE` loss.

```
Input → (time_steps=60, n_features=14)
         │
┌────────┴──────────────────────────────────────┐
│  GRU (best)          │ Stacked LSTM │ BiLSTM  │
│  GRU(64, seq=True)   │ LSTM(64,T)   │ BiLSTM  │
│  Dropout(0.2)        │ Dropout(0.2) │ (64,T)  │
│  GRU(64)             │ LSTM(64)     │ BiLSTM  │
│                      │              │ (64)    │
└──────────────────────┴──────────────┴─────────┘
         │
    Dropout(0.2)
    Dense(32, relu)
    Dense(1)  →  predicted Close price
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Time steps (window) | 60 |
| Train / Val / Test | 72% / 8% / 20% |
| Max epochs | 50 |
| Batch size | 32 |
| Early stopping patience | 10 epochs |
| LR reduction factor | 0.5 (patience=5) |
| Scaler | MinMaxScaler [0, 1] |

---

## 🔧 Feature Engineering

| Feature | Description |
|---------|-------------|
| `MA10` | 10-day simple moving average |
| `MA50` | 50-day simple moving average |
| `EMA20` | 20-day exponential moving average |
| `RSI14` | 14-day Relative Strength Index |
| `Lag_1/2/3/5/10` | Previous close prices (5 lags) |
| `Return_1d` | Daily percentage return |

---

## 🖥️ Streamlit Deployment

The app provides a 4-tab interface:

| Tab | Content |
|-----|---------|
| 📊 **EDA** | Price history with MA overlays, volume bar chart, RSI panel, correlation heatmap |
| 🔧 **Features** | Moving averages chart, return distribution histogram, feature table |
| 🚀 **Train** | Live epoch-by-epoch loss readout, progress bar, model selection |
| 📈 **Results** | RMSE/MAE cards, comparison table, actual vs predicted chart, CSV export |

**Sidebar controls** let you select models, tune hyperparameters (time steps, epochs, batch size, train split), and upload any compatible CSV — without touching code.

---

## 📦 Dependencies

```
streamlit>=1.32.0
tensorflow>=2.15.0
scikit-learn>=1.4.0
pandas>=2.1.0
numpy>=1.26.0
plotly>=5.19.0
matplotlib>=3.8.0
seaborn>=0.13.0
```

---

## 🗺️ Roadmap

- [ ] Attention mechanism over GRU output sequence
- [ ] Optuna hyperparameter search (target RMSE < $10)
- [ ] Exogenous features: VIX index, S&P 500 return
- [ ] Multi-step ahead forecasting (5-day / 10-day horizon)
- [ ] Dockerized deployment
- [ ] REST API with FastAPI + model versioning via MLflow

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙋 Author

Built as a portfolio-grade end-to-end ML project demonstrating production-level time-series forecasting with deep learning.

If you found this useful, give it a ⭐ on GitHub!
