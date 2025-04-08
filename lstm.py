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
data = df[["Price_Monthly_Avg", "TAVG_Monthly_Avg"]].fillna(0)

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
train_cutoff = pd.Timestamp("2024-07-31")
test_start = pd.Timestamp("2024-08-31")

train_mask = y_dates <= train_cutoff
test_mask = y_dates >= test_start

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]
test_dates = y_dates[test_mask]

# 6. Build LSTM model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(seq_len, X.shape[2])),
    # Dropout(0.3),
    LSTM(64),
    # Dropout(0.3),
    Dense(1)
])
model.compile(loss='mse', optimizer='adam')
model.summary()

# 7. Train
history = model.fit(X_train, y_train, epochs=100, batch_size=16,
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

# === 12. Forecast future prices recursively (Next 4 months) ===
future_months = 4
last_sequence = scaled_data[-seq_len:].copy()
future_predictions_scaled = []

for _ in range(future_months):
    input_seq = last_sequence.reshape(1, seq_len, scaled_data.shape[1])
    pred = model.predict(input_seq, verbose=0)[0][0]

    # Next row for sequence (predicted price, 0 for TAVG)
    next_row = np.zeros((scaled_data.shape[1],))
    next_row[0] = pred
    last_sequence = np.vstack((last_sequence[1:], next_row))

    future_predictions_scaled.append(pred)

# === 13. Inverse transform future predictions ===
dummy_future = np.zeros((future_months, scaled_data.shape[1]))
dummy_future[:, 0] = future_predictions_scaled
future_prices = scaler.inverse_transform(dummy_future)[:, 0]

# === 14. Future date index ===
last_known_date = df.index[-1]
future_dates = pd.date_range(start=last_known_date + pd.DateOffset(months=1), periods=future_months, freq='MS')

# === 15. Plot (Full Timeline + Forecast) ===
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['Price_Monthly_Avg'], label='Actual (Full Timeline)', color='blue')
plt.plot(test_dates, y_pred, label='Predicted (Last 4 Months)', color='orange', marker='o')
plt.plot(future_dates, future_prices, label='Forecast (Next 4 Months)', color='green', linestyle='--', marker='o')
plt.plot([test_dates[-1], future_dates[0]], [y_pred[-1], future_prices[0]], color='green', linestyle='--')
plt.title("LSTM Cocoa Price Forecast (Full Timeline)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# === 16. Plot (Zoomed In: Last 4 Months + Future 4 Months) ===
plt.figure(figsize=(14, 6))
plt.plot(test_dates, y_actual, label='Actual (Last 4 Months)', color='blue')
plt.plot(test_dates, y_pred, label='Predicted (Last 4 Months)', color='orange', marker='o')
plt.plot(future_dates, future_prices, label='Forecast (Next 4 Months)', color='green', linestyle='--', marker='o')
plt.plot([test_dates[-1], future_dates[0]], [y_pred[-1], future_prices[0]], color='green', linestyle='--')
plt.title("LSTM Cocoa Price Forecast (Zoomed: Last 4M + Next 4M)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
