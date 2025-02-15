import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the dataset
data_path = 'Equites_Historical_Adjusted_Prices_Report.csv'
df = pd.read_csv(data_path)

# Convert 'Date' to datetime and sort values
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['Symbol', 'Date'])

# Select relevant columns for analysis
data = df[['Symbol', 'Date', 'Close', 'High', 'Low', 'Change', '% Change', 'Volume Traded', 'Value Traded (SAR)']]

# Ensure there are no NaN or infinite values in the selected columns
columns_to_check = ['High', 'Low', 'Change', '% Change', 'Volume Traded', 'Value Traded (SAR)', 'Close']

# Replace NaN values with 0 (or other strategies like using the mean)
data[columns_to_check] = data[columns_to_check].fillna(0)

# Replace infinity values with 0
data[columns_to_check] = data[columns_to_check].replace([np.inf, -np.inf], 0)

# Normalize the 'Close' prices, 'High', 'Low', 'Change', '% Change', 'Volume Traded', and 'Value Traded (SAR)'
scalers = {}

# Function to create sequences
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, :])
        y.append(data[i, 0])  # Predicting 'Close' price
    return np.array(X), np.array(y)

sequence_length = 60
X_all, y_all = [], []

# Prepare data for all companies
for symbol in data['Symbol'].unique():
    symbol_data = data[data['Symbol'] == symbol][['Date', 'Close', 'High', 'Low', 'Change', '% Change', 'Volume Traded', 'Value Traded (SAR)']].set_index('Date')
    
    # Normalize the selected columns
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(symbol_data[columns_to_check].values)
    scalers[symbol] = scaler
    
    # Create sequences for LSTM model
    X, y = create_sequences(data_scaled, sequence_length)
    
    if X.shape[0] > 0:
        X_all.append(X)
        y_all.append(y)

X_all = np.vstack(X_all) if X_all else np.array([])
y_all = np.concatenate(y_all) if y_all else np.array([])

if X_all.size == 0 or y_all.size == 0:
    raise ValueError("No valid data to train the model. Please check the data preparation step.")

# TimeSeriesSplit for robust splitting
ts_split = TimeSeriesSplit(n_splits=5)
for train_index, test_index in ts_split.split(X_all):
    X_train, X_test = X_all[train_index], X_all[test_index]
    y_train, y_test = y_all[train_index], y_all[test_index]

# Reshape data to 3D (required for LSTM input)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Callbacks: Early stopping and learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch > 5:
        return lr * 0.5
    return lr

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    LearningRateScheduler(lr_scheduler)
]

# Train the model
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=50,
    validation_data=(X_test, y_test),
    callbacks=callbacks
)

# Save the trained model
model.save('stock_price_model_with_volume.h5')

# Plotting the loss graph
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Predict on the test set
predictions = model.predict(X_test)

# Calculate residuals
residuals = y_test - predictions.flatten()

# Plot residuals
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=50, alpha=0.75, label='Residuals')
plt.axvline(0, color='red', linestyle='--', label='Zero Error')
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

# Calculate metrics
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Evaluation Results:")
print(f'Mean Squared Error: {mse:.4f}')
print(f'Mean Absolute Error: {mae:.4f}')
print(f'R-squared: {r2:.4f}')

# Plot actual vs predicted prices
plt.figure(figsize=(8, 6))
plt.plot(y_test, label='Actual Prices', marker='o')
plt.plot(predictions, label='Predicted Prices', marker='x')
plt.title('Actual vs Predicted Prices')
plt.xlabel('Sample')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

#2 Features close and Volume (the file on github)
#Evaluation Results:
#Mean Squared Error: 0.0006
#Mean Absolute Error: 0.0162
#R-squared: 0.9896

# all the Features (the file rn on the desktop not on the folder and it's name (with all Features)) 
# Evaluation Results:
#Mean Squared Error: 0.0006
#Mean Absolute Error: 0.0124
#R-squared: 0.9898

#latest (name (with vol) and it's now on the same folder)
#Evaluation Results:
#Mean Squared Error: 0.0006
#Mean Absolute Error: 0.0123
#R-squared: 0.9901
