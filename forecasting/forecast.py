import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit as st

# -----------------------------
# Sequence creation
# -----------------------------
def create_sequences(data, seq_len=10):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)


# -----------------------------
# LSTM Model
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 50, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# -----------------------------
# Load & preprocess data
# -----------------------------
# forecast.py

# -----------------------------
# Load & preprocess data (accept DataFrame)
# -----------------------------
def load_and_preprocess(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Ensure date column exists
    if 'date' not in df.columns:
        raise ValueError("DataFrame must have a 'date' column")

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date')

    # Fill missing sales
    df['sales'] = df['sales'].ffill()

    # Limit data to prevent memory crash
    df = df.tail(1000)  # last 1000 rows for LSTM

    values = df['sales'].values.reshape(-1, 1)
    return df, values

@st.cache_resource
def train_model(df):
    data, values = load_and_preprocess(df)

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values)

    X, y = create_sequences(scaled_values)

    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)

    model = LSTMModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(10):
        output = model(X)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(X)

    y_true = y.numpy()
    y_pred = y_pred.numpy()

    y_true_inv = scaler.inverse_transform(y_true)
    y_pred_inv = scaler.inverse_transform(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
    mae = mean_absolute_error(y_true_inv, y_pred_inv)

    return model, scaler, rmse, mae


def predict_next_day(df):
    model, scaler, rmse, mae = train_model(df)

    sales = df['sales'].values
    if len(sales) < 10:
        raise ValueError("Need at least 10 rows of sales data")

    last_10 = sales[-10:].reshape(-1, 1)
    last_10_scaled = scaler.transform(last_10)

    X_input = torch.FloatTensor(last_10_scaled).view(1, 10, 1)

    model.eval()
    with torch.no_grad():
        prediction_scaled = model(X_input).numpy()

    prediction = scaler.inverse_transform(prediction_scaled)
    return prediction[0][0], rmse, mae