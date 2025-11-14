# =============================================
# AI Traffic Congestion Prediction System
# For ASEAN Urban Mobility Challenge
# TechSkills Challenge Submission
# =============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("AI Traffic Prediction System Initializing...")
print("============================================")

def generate_traffic_dataset():
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-02-29', freq='H')
    congestion_levels = []
    
    for date in dates:
        hour = date.hour
        is_weekday = date.weekday() < 5
        
        if is_weekday:
            if 7 <= hour <= 9:
                base_level = 82
            elif 16 <= hour <= 19:
                base_level = 85
            elif 12 <= hour <= 13:
                base_level = 65
            else:
                base_level = 45
        else:
            if 11 <= hour <= 17:
                base_level = 70
            else:
                base_level = 38
        
        noise = np.random.normal(0, 6)
        congestion = base_level + noise
        congestion = max(25, min(95, congestion))
        congestion_levels.append(congestion)
    
    return pd.DataFrame({
        'timestamp': dates,
        'congestion_percentage': congestion_levels
    })

print("Generating traffic dataset...")
traffic_df = generate_traffic_dataset()
print(f"Dataset created: {len(traffic_df)} hourly records")

plt.figure(figsize=(12, 4))
plt.plot(traffic_df['timestamp'], traffic_df['congestion_percentage'], 
         linewidth=0.8, color='#2E86AB')
plt.title('Traffic Congestion Patterns - ASEAN Metropolitan Area')
plt.xlabel('Date')
plt.ylabel('Congestion Level (%)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('traffic_patterns.png')
plt.show()

def create_sequences(data, sequence_length=24):
    features, targets = [], []
    for i in range(len(data) - sequence_length):
        features.append(data[i:(i + sequence_length)])
        targets.append(data[i + sequence_length])
    return np.array(features), np.array(targets)

scaler = MinMaxScaler(feature_range=(0, 1))
normalized_data = scaler.fit_transform(traffic_df[['congestion_percentage']])

SEQUENCE_LENGTH = 24
X, y = create_sequences(normalized_data, SEQUENCE_LENGTH)

split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print(f"Training sequences: {len(X_train)}")
print(f"Testing sequences: {len(X_test)}")

print("Building predictive model...")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    
    print("Using LSTM Neural Network architecture")
    
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 1)),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    
    print("Training model...")
    training_history = model.fit(
        X_train, y_train,
        epochs=60,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1,
        shuffle=False
    )
    
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
except ImportError:
    print("TensorFlow not available - Using Random Forest")
    from sklearn.ensemble import RandomForestRegressor
    
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42
    )
    
    model.fit(X_train_flat, y_train)
    train_predictions = model.predict(X_train_flat)
    test_predictions = model.predict(X_test_flat)

train_predictions_actual = scaler.inverse_transform(train_predictions.reshape(-1, 1))
test_predictions_actual = scaler.inverse_transform(test_predictions.reshape(-1, 1))
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

train_mae = mean_absolute_error(y_train_actual, train_predictions_actual)
test_mae = mean_absolute_error(y_test_actual, test_predictions_actual)
train_r2 = r2_score(y_train_actual, train_predictions_actual)
test_r2 = r2_score(y_test_actual, test_predictions_actual)

print("\n" + "="*60)
print("MODEL PERFORMANCE EVALUATION")
print("="*60)
print(f"Training Mean Absolute Error: {train_mae:.2f}%")
print(f"Testing Mean Absolute Error: {test_mae:.2f}%")
print(f"Training R-squared: {train_r2:.4f}")
print(f"Testing R-squared: {test_r2:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0,0].plot(y_train_actual[:150], label='Actual', color='#2E86AB', linewidth=1.5)
axes[0,0].plot(train_predictions_actual[:150], label='Predicted', color='#A23B72', linewidth=1.5)
axes[0,0].set_title('Model Performance - Training Data')
axes[0,0].set_xlabel('Time Sequence')
axes[0,0].set_ylabel('Congestion (%)')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(y_test_actual[:100], label='Actual', color='#2E86AB', linewidth=1.5)
axes[0,1].plot(test_predictions_actual[:100], label='Predicted', color='#A23B72', linewidth=1.5)
axes[0,1].set_title('Model Performance - Testing Data')
axes[0,1].set_xlabel('Time Sequence')
axes[0,1].set_ylabel('Congestion (%)')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

axes[1,0].scatter(y_test_actual, test_predictions_actual, alpha=0.6, color='#F18F01')
axes[1,0].plot([20, 95], [20, 95], 'k--', linewidth=1, label='Ideal Prediction')
axes[1,0].set_xlabel('Actual Congestion (%)')
axes[1,0].set_ylabel('Predicted Congestion (%)')
axes[1,0].set_title('Prediction Accuracy Analysis')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

prediction_errors = y_test_actual.flatten() - test_predictions_actual.flatten()
axes[1,1].hist(prediction_errors, bins=30, color='#C73E1D', alpha=0.7)
axes[1,1].axvline(x=0, color='black', linestyle='--', linewidth=1)
axes[1,1].set_xlabel('Prediction Error (%)')
axes[1,1].set_ylabel('Frequency')
axes[1,1].set_title('Distribution of Prediction Errors')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("TRAFFIC PATTERN ANALYSIS")
print("="*60)

traffic_df['hour'] = traffic_df['timestamp'].dt.hour
hourly_patterns = traffic_df.groupby('hour')['congestion_percentage'].mean()
peak_hour = hourly_patterns.idxmax()
peak_congestion = hourly_patterns.max()

print(f"Peak congestion hour: {peak_hour}:00 ({peak_congestion:.1f}%)")
print(f"Morning rush (7-9 AM): {hourly_patterns[7:10].mean():.1f}%")
print(f"Evening rush (5-7 PM): {hourly_patterns[17:19].mean():.1f}%")

print("\n" + "="*60)
print("24-HOUR TRAFFIC FORECAST")
print("="*60)

latest_sequence = normalized_data[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, 1)
forecast = []

for hour in range(24):
    if 'tf' in locals():
        next_pred = model.predict(latest_sequence, verbose=0)
        forecast.append(next_pred[0, 0])
        latest_sequence = np.roll(latest_sequence, -1, axis=1)
        latest_sequence[0, -1, 0] = next_pred[0, 0]
    else:
        next_pred = model.predict(latest_sequence.reshape(1, -1))
        forecast.append(next_pred[0])
        latest_sequence = np.roll(latest_sequence, -1)
        latest_sequence[0, -1] = next_pred[0]

forecast_actual = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

start_hour = traffic_df['timestamp'].iloc[-1].hour
for i, prediction in enumerate(forecast_actual):
    forecast_hour = (start_hour + i + 1) % 24
    time_label = f"{forecast_hour:02d}:00"
    print(f"{time_label}: {prediction[0]:.1f}% congestion")

plt.figure(figsize=(12, 6))
hours = [(start_hour + i + 1) % 24 for i in range(24)]
plt.plot(hours, forecast_actual, marker='o', linewidth=2, color='#2E86AB')
plt.fill_between(hours, forecast_actual.flatten() - 5, forecast_actual.flatten() + 5, 
                 alpha=0.2, color='#2E86AB')
plt.title('24-Hour Traffic Congestion Forecast')
plt.xlabel('Hour of Day')
plt.ylabel('Predicted Congestion (%)')
plt.grid(True, alpha=0.3)
plt.xticks(range(0, 24, 2))
plt.ylim(20, 95)
plt.tight_layout()
plt.savefig('24_hour_forecast.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("SYSTEM EXECUTION COMPLETED SUCCESSFULLY")
print("="*60)