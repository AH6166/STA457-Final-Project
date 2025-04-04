# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/16eKeL4W2y4sQ-KlylmJ1MHXJ6Ebl56BM
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv("price_climate.csv", parse_dates=["date"])
df.set_index("date", inplace=True)

# 2. Select features
data = df[["Price_Monthly_Avg", "PRCP_Monthly_Avg", "TAVG_Monthly_Avg"]].fillna(0)

# 3. Normalize features
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 4. Create sequences
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i, 0])  # predict price only
    return np.array(X), np.array(y)

seq_len = 18  # use past 12 months
X, y = create_sequences(scaled_data, seq_len)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Test size: {len(y_test)}")
print(len(X))
print(len(X)*0.8)
print(len(df))

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(seq_len, X.shape[2])),
    LSTM(32),
    Dense(1)
])

model.compile(loss='mse', optimizer='adam')
model.summary()

# Train
history = model.fit(X_train, y_train, epochs=50, batch_size=16,
                    validation_data=(X_test, y_test), verbose=1)

# Predict
y_pred_scaled = model.predict(X_test)

# Rebuild full predicted series (merge back with rest of data for inverse transform)
dummy = np.zeros((len(y_pred_scaled), scaled_data.shape[1]))
dummy[:, 0] = y_pred_scaled[:, 0]  # only price column predicted

# Inverse transform
y_pred = scaler.inverse_transform(dummy)[:, 0]
y_actual = scaler.inverse_transform(np.concatenate([
    y_test.reshape(-1, 1),  # <-- Put price in the correct position
    np.zeros((len(y_test), scaled_data.shape[1]-1))
], axis=1))[:, 0]  # <-- Get the first column (price)

from sklearn.metrics import mean_squared_error, mean_absolute_error

rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
mae = mean_absolute_error(y_actual, y_pred)
mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100

print(f"✅ RMSE: {rmse:.2f}")
print(f"✅ MAE: {mae:.2f}")
print(f"✅ MAPE: {mape:.2f}%")

plt.figure(figsize=(10, 5))
plt.plot(y_actual, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title("LSTM Forecast of Cocoa Prices")
plt.xlabel("Time (months)")
plt.ylabel("Price")
plt.legend()

plt.tight_layout()
plt.show()


# Assume this is your date Series from the dataset
# Replace this with your actual date column
# For example: dates = pd.to_datetime(df['date_column'])
dates = pd.date_range(start="2018-11", periods=len(y_actual), freq='M')

# Now plot using the dates
plt.figure(figsize=(12, 6))
plt.plot(dates, y_actual, label='Actual')
plt.plot(dates, y_pred, label='Predicted')

plt.title("LSTM Forecast of Cocoa Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()

# Format the x-axis to show year-month
plt.gcf().autofmt_xdate()  # Rotate date labels
plt.tight_layout()
plt.show()

print(len(y_actual))

# Start from the last sequence in the dataset
last_sequence = scaled_data[-seq_len:].copy()

future_predictions_scaled = []

for _ in range(12):
    input_seq = last_sequence.reshape(1, seq_len, scaled_data.shape[1])
    pred = model.predict(input_seq, verbose=0)[0][0]

    # Build next row with predicted price, and dummy 0s for PRCP and TAVG
    next_row = np.zeros((scaled_data.shape[1],))
    next_row[0] = pred  # predicted price

    # Append prediction to list
    future_predictions_scaled.append(pred)

    # Update sequence: drop first, add new predicted row
    last_sequence = np.vstack((last_sequence[1:], next_row))

# Reconstruct dummy for inverse transform
dummy_future = np.zeros((len(future_predictions_scaled), scaled_data.shape[1]))
dummy_future[:, 0] = future_predictions_scaled

# Inverse transform to get price values
future_prices = scaler.inverse_transform(dummy_future)[:, 0]

# Get the last date from the original dataset
last_y_pred_date = df.index[seq_len + split + len(y_pred) - 1]
future_dates = pd.date_range(start=last_y_pred_date + pd.DateOffset(months=1), periods=12, freq='M')

print(df)
print(last_y_pred_date)
print(future_dates)

plt.figure(figsize=(12, 6))

# Plot existing predictions with actual values
plt.plot(dates, y_actual, label='Actual')
plt.plot(dates, y_pred, label='Predicted')

# Plot future predictions
plt.plot(future_dates, future_prices, label='Future Forecast', linestyle='--', marker='o')

plt.title("LSTM Forecast of Cocoa Prices (Including 12-Month Future Forecast)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.show()
print(dates)

plt.figure(figsize=(12, 6))

# Plot actual and predicted
plt.plot(dates, y_actual, label='Actual')
plt.plot(dates, y_pred, label='Predicted')

# Fix the future dates starting right after last predicted month
last_y_pred_date = df.index[seq_len + split + len(y_pred) - 1]
future_dates = pd.date_range(start=last_y_pred_date + pd.DateOffset(months=1), periods=12, freq='M')

# Plot future predictions
plt.plot(future_dates, future_prices, label='Future Forecast', linestyle='--', marker='o', color='green')

plt.title("LSTM Forecast of Cocoa Prices (Including 12-Month Future Forecast)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.show()

print(dates)
print(last_y_pred_date)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.layers import Dropout

# 1. Load data
df = pd.read_csv("price_climate.csv", parse_dates=["date"])
df.set_index("date", inplace=True)

# 2. Select features
data = df[["Price_Monthly_Avg", "PRCP_Monthly_Avg", "TAVG_Monthly_Avg"]].fillna(0)

# 3. Normalize features
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 4. Create sequences WITH date tracking
def create_sequences_with_dates(data, dates, seq_len):
    X, y, y_dates = [], [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i, 0])
        y_dates.append(dates[i])
    return np.array(X), np.array(y), np.array(y_dates)

seq_len = 6
dates = data.index  # datetime index
X, y, y_dates = create_sequences_with_dates(scaled_data, dates, seq_len)

# 5. Time-based split
train_cutoff = pd.Timestamp("2018-10-31")
test_start = pd.Timestamp("2018-11-30")

train_mask = y_dates <= train_cutoff
test_mask = y_dates >= test_start

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]
test_dates = y_dates[test_mask]

# 6. Build LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(seq_len, X.shape[2])),
    # Dropout(0.2),
    LSTM(32),
    # Dropout(0.2),
    Dense(1)
])
model.compile(loss='mse', optimizer='adam')
model.summary()

# 7. Train
history = model.fit(X_train, y_train, epochs=50, batch_size=16,
                    validation_data=(X_test, y_test), verbose=1)

# 8. Predict (on test set)
y_pred_scaled = model.predict(X_test)

# 9. Inverse transform
# For prediction: construct dummy to match shape
dummy_pred = np.zeros((len(y_pred_scaled), scaled_data.shape[1]))
dummy_pred[:, 0] = y_pred_scaled[:, 0]
y_pred = scaler.inverse_transform(dummy_pred)[:, 0]

# For actual: same for y_test
dummy_actual = np.zeros((len(y_test), scaled_data.shape[1]))
dummy_actual[:, 0] = y_test
y_actual = scaler.inverse_transform(dummy_actual)[:, 0]

# 10. Metrics
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
mae = mean_absolute_error(y_actual, y_pred)
mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
print(f"✅ RMSE: {rmse:.2f}")
print(f"✅ MAE: {mae:.2f}")
print(f"✅ MAPE: {mape:.2f}%")

# print(f"✅ ME (Mean Error): {me:.2f}")


# 11. Plot
plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_actual, label='Actual')
plt.plot(test_dates, y_pred, label='Predicted')
plt.title("LSTM Forecast of Cocoa Prices (Test: From 2018-11-30)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

me = np.mean(y_pred - y_actual)
abs_me = abs(me)
print(f"✅ ME (Mean Error): {abs_me:.2f}")

# Start with the last known sequence
last_sequence = scaled_data[-seq_len:].copy()  # shape: (18, 3)

future_predictions_scaled = []

for _ in range(6):
    input_seq = last_sequence.reshape(1, seq_len, scaled_data.shape[1])
    pred = model.predict(input_seq, verbose=0)[0][0]

    # Prepare next input row: predicted price + dummy 0s for other features
    next_row = np.zeros((scaled_data.shape[1],))
    next_row[0] = pred

    # Append to sequence and slide window
    last_sequence = np.vstack((last_sequence[1:], next_row))
    future_predictions_scaled.append(pred)

# Inverse transform predicted values
dummy_future = np.zeros((6, scaled_data.shape[1]))
dummy_future[:, 0] = future_predictions_scaled
future_prices = scaler.inverse_transform(dummy_future)[:, 0]

last_known_date = df.index[-1]
future_dates = pd.date_range(start=last_known_date + pd.DateOffset(months=1), periods=6, freq='M')

plt.figure(figsize=(12, 6))

# Plot previous predictions
plt.plot(test_dates, y_actual, label='Actual')
plt.plot(test_dates, y_pred, label='Predicted')

# Plot future predictions
plt.plot(future_dates, future_prices, label='Future Forecast (Next 12 Months)', linestyle='--', marker='o', color='green')

plt.title("LSTM Forecast of Cocoa Prices (Including 12-Month Future Forecast)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.layers import Dropout

# 1. Load data
df = pd.read_csv("price_climate.csv", parse_dates=["date"])
df.set_index("date", inplace=True)

# 2. Select features
data = df[["Price_Monthly_Avg", "PRCP_Monthly_Avg", "TAVG_Monthly_Avg"]].fillna(0)

# 3. Normalize features
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 4. Create sequences WITH date tracking
def create_sequences_with_dates(data, dates, seq_len):
    X, y, y_dates = [], [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i, 0])
        y_dates.append(dates[i])
    return np.array(X), np.array(y), np.array(y_dates)

seq_len = 12
dates = data.index  # datetime index
X, y, y_dates = create_sequences_with_dates(scaled_data, dates, seq_len)

# 5. Time-based split
train_cutoff = pd.Timestamp("2018-10-31")
test_start = pd.Timestamp("2018-11-30")

train_mask = y_dates <= train_cutoff
test_mask = y_dates >= test_start

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]
test_dates = y_dates[test_mask]

# 6. Build LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(seq_len, X.shape[2])),
    # Dropout(0.3),
    LSTM(32),
    # Dropout(0.3),
    Dense(1)
])
model.compile(loss='mse', optimizer='adam')
model.summary()

# 7. Train
history = model.fit(X_train, y_train, epochs=50, batch_size=16,
                    validation_data=(X_test, y_test), verbose=1)

# 8. Predict (on test set)
y_pred_scaled = model.predict(X_test)

# 9. Inverse transform
# For prediction: construct dummy to match shape
dummy_pred = np.zeros((len(y_pred_scaled), scaled_data.shape[1]))
dummy_pred[:, 0] = y_pred_scaled[:, 0]
y_pred = scaler.inverse_transform(dummy_pred)[:, 0]

# For actual: same for y_test
dummy_actual = np.zeros((len(y_test), scaled_data.shape[1]))
dummy_actual[:, 0] = y_test
y_actual = scaler.inverse_transform(dummy_actual)[:, 0]

# 10. Metrics
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
mae = mean_absolute_error(y_actual, y_pred)
mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
print(f"✅ RMSE: {rmse:.2f}")
print(f"✅ MAE: {mae:.2f}")
print(f"✅ MAPE: {mape:.2f}%")

# print(f"✅ ME (Mean Error): {me:.2f}")


# 11. Plot
plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_actual, label='Actual')
plt.plot(test_dates, y_pred, label='Predicted')
plt.title("LSTM Forecast of Cocoa Prices (Test: From 2018-11-30)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Start with the last known sequence
last_sequence = scaled_data[-seq_len:].copy()  # shape: (18, 3)

future_predictions_scaled = []

for _ in range(6):
    input_seq = last_sequence.reshape(1, seq_len, scaled_data.shape[1])
    pred = model.predict(input_seq, verbose=0)[0][0]

    # Prepare next input row: predicted price + dummy 0s for other features
    next_row = np.zeros((scaled_data.shape[1],))
    next_row[0] = pred

    # Append to sequence and slide window
    last_sequence = np.vstack((last_sequence[1:], next_row))
    future_predictions_scaled.append(pred)

# Inverse transform predicted values
dummy_future = np.zeros((6, scaled_data.shape[1]))
dummy_future[:, 0] = future_predictions_scaled
future_prices = scaler.inverse_transform(dummy_future)[:, 0]
last_known_date = df.index[-1]
future_dates = pd.date_range(start=last_known_date, periods=6, freq='M')
#  + pd.DateOffset(months=1)
plt.figure(figsize=(12, 6))

# Plot previous predictions
plt.plot(test_dates, y_actual, label='Actual')
plt.plot(test_dates, y_pred, label='Predicted')

# Plot future predictions
plt.plot(future_dates, future_prices, label='Future Forecast (Next 12 Months)', linestyle='--', marker='o', color='green')

plt.title("LSTM Forecast of Cocoa Prices (Including 12-Month Future Forecast)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

residuals = y_actual - y_pred
plt.figure(figsize=(12, 4))
plt.plot(test_dates, residuals, label="Residuals")
plt.axhline(y=0, color='red', linestyle='--')
plt.title("Residual Plot")
plt.xlabel("Date")
plt.ylabel("Residual (Actual - Predicted)")
plt.legend()
plt.tight_layout()
plt.show()
from statsmodels.graphics.tsaplots import plot_acf

plt.figure(figsize=(10, 4))
plot_acf(residuals, lags=20)
plt.title("ACF of Residuals")
plt.tight_layout()
plt.show()
from statsmodels.stats.diagnostic import acorr_ljungbox

ljung_box_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
print(ljung_box_result)