import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import yfinance as yf
import threading
import time
# import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests from your React app

# Load the dataset and model
data_path = os.path.join(os.path.dirname(__file__), 'Equites_Historical_Adjusted_Prices_Report.csv')
df = pd.read_csv(data_path)

model_path = os.path.join(os.path.dirname(__file__), 'stock_price_model_V4.h5')
model = tf.keras.models.load_model(model_path)

available_symbols = df['Symbol'].unique()
scalers = {}

cache = []  # Cached results


def fetch_stock_data(symbol, period="6mo", interval="1d"):
    ticker = str(symbol) + ".SR"

    data = yf.download(ticker, period=period, interval=interval, group_by='ticker')
    data.columns = data.columns.droplevel(0)
    data.index.name = "Date"
    data.columns.name = None
    data = data

    # Calculate daily change and percentage change
    data['Change'] = data['Close'].diff()
    data['% Change'] = (data['Close'].pct_change()) * 100
    data['Value Traded (SAR)'] = (data['Close'] + data['Open'])//2 * data['Volume']
    data = data[['Close', 'High', 'Low', 'Change', '% Change', 'Volume', 'Value Traded (SAR)']]
    data = data.rename(columns={'Volume': 'Volume Traded'})
    data = data.iloc[1:]

  
    return data

def fetch_company_data(symbol):
    ticker = str(symbol) + ".SR"
    stock_data = yf.Ticker(ticker)
    return stock_data.info

def format_number(number):
    if number >= 1e9:
        return f"{round(number / 1e9, 1)}B"
    elif number >= 1e6:
        return f"{round(number / 1e6, 1)}M"
    elif number >= 1e3:
        return f"{round(number / 1e3, 1)}K"
    else:
        return str(round(number,1))


def predict_stock(symbol, sequence_length=60):

    stock_data = fetch_stock_data(symbol)

    if stock_data is None or stock_data.empty:
        print(f"No data found for {symbol}.")
        return None, None, None

    if len(stock_data) < sequence_length:
        print(f"Not enough data for ticker: {symbol}.")
        return None, None, None

    current_price = stock_data['Close'].iloc[-1]
    last_day_price = stock_data['Close'].iloc[-2]

    # Select the same features used during training
    feature_columns = ['Close', 'High', 'Low', 'Change', '% Change', 'Volume Traded', 'Value Traded (SAR)']

    # Check if all required columns exist
    for col in feature_columns:
        if col not in stock_data.columns:
            print(f"Missing column: {col} in stock data for {symbol}")
            return None, None, None

    # Normalize data (Recreate the scaler)
    scaler = MinMaxScaler(feature_range=(0, 1))
    stock_data_scaled = scaler.fit_transform(stock_data[feature_columns].values)

    # Extract the last sequence for prediction
    last_sequence = stock_data_scaled[-sequence_length:]
    last_sequence = np.array(last_sequence).reshape(1, sequence_length, len(feature_columns))

    # Predict next-day stock price
    next_day_prediction = model.predict(last_sequence)[0][0]

    # Reverse scaling (only for 'Close' price)
    predicted_price = scaler.inverse_transform([[next_day_prediction] + [0] * (len(feature_columns) - 1)])[0][0]

    return last_day_price, current_price, predicted_price




company_info_path = os.path.join(os.path.dirname(__file__), 'company_info.csv')
company_info_df = pd.read_csv(company_info_path)
companies = [
    {"id": idx + 1, "name": row["Company Name"], "symbol": row["Symbol"]}
    for idx, row in company_info_df.iterrows()
]

def refresh_cache():
    """Pre-compute and store data for all companies in the cache."""
    global cache
    cache = []  # Clear the existing cache

    for company in companies:
        last_day_price , current_price, predicted_price = predict_stock(company['symbol'])
        cache.append({
            "id": company["id"],
            "name": company["name"],
            "symbol": company["symbol"],
            "current_price": round(current_price, 2) if current_price else None,
            "predicted_price": round(predicted_price, 2) if predicted_price else None,
            "last_day_price": last_day_price
        })

    print("Cache refreshed!")


@app.route('/api/companies', methods=['GET'])
def get_companies():
    """Endpoint to fetch pre-computed company data."""
    return jsonify(cache)

@app.route('/api/company/<int:company_id>', methods=['GET'])
def get_company_details(company_id):
    company = next((c for c in companies if c["id"] == company_id), None)
    if not company:
        return jsonify({"error": "Company not found"}), 404

    try:
        last_day_price , current_price, predicted_price = predict_stock(company['symbol'])
        stock_data = fetch_stock_data(company['symbol'])
        historical_prices = stock_data["Close"]
        company_data = fetch_company_data(company["symbol"])

        # Get historical dates
        historical_dates = historical_prices.index.strftime('%Y-%m-%d').tolist()
        
        # Generate future dates (next 30 days after last historical date)
        last_date = historical_prices.index[-1]
        future_dates = pd.date_range(
            start=last_date,
            periods=2
        ).strftime('%Y-%m-%d').tolist()

        # Create recommendation
        recommendation = "Buy" if predicted_price > current_price else "Sell" if predicted_price < current_price else "Hold"

        return jsonify({
            "id": company_id,
            "name": company["name"],
            "symbol": company["symbol"],
            "market_cap": format_number(company_data["marketCap"]),
            "volume": format_number(company_data["volume"]),  
            "sector": company_data.get("sector"),
            "recommendation": company_data.get("recommendationKey"),
            "historical": {
                "dates": historical_dates[-30:],
                "prices": historical_prices[-30:].tolist()
            },
            "forecast": {
                "dates": future_dates,
                "prices": [current_price ,round(predicted_price, 2)]
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def start_cache_refresh(interval=3600):
    """Periodically refresh the cache in a separate thread."""
    def refresh_loop():
        while True:
            refresh_cache()
            time.sleep(interval)

    thread = threading.Thread(target=refresh_loop, daemon=True)
    thread.start()


if __name__ == '__main__':
    # refresh_cache()  # Initial cache refresh
    start_cache_refresh(interval=3600)  # Refresh cache every hour
    app.run(debug=True)


# # Helper functions
# def create_sequences(data, sequence_length):
#     X, y = [], []
#     for i in range(sequence_length, len(data)):
#         X.append(data[i-sequence_length:i, 0])
#         y.append(data[i, 0])
#     return np.array(X), np.array(y)


# def fetch_stock_data(symbol):
#     """
#     Fetch the last 30 days of stock data for the given ticker.
#     """
#     ticker = str(symbol) + ".SR"
#     stock_data = yf.Ticker(ticker)
#     data = stock_data.history(period='3mo')
    
#     if data.empty:
#         raise ValueError(f"No data found for ticker: {ticker}")
    
#     return data[['Close']]

# def predict_stock(symbol, sequence_length=7):
#     """Predict the next stock price using the LSTM model."""
#     try:
#         stock_data = fetch_stock_data(symbol)
#     except ValueError as e:
#         print(str(e))
#         return None, None

#     # Get the most recent price
#     current_price = stock_data['Close'].iloc[-1]

#     # Check if there are enough days for the sequence
#     if len(stock_data) < sequence_length:
#         print(f"Not enough data for ticker: {symbol}.")
#         return current_price, None

#     # Scale the data
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     data_scaled = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))
#     scalers[symbol] = scaler

#     # Get the last sequence for prediction
#     last_sequence = data_scaled[-sequence_length:]
#     last_sequence = last_sequence.reshape(1, sequence_length, 1)  # Reshape for LSTM input

#     # Predict the next price
#     next_day_prediction = model.predict(last_sequence)[0][0]

#     # Convert back to original scale
#     predicted_price = scaler.inverse_transform([[next_day_prediction]])[0][0]

#     return current_price, predicted_price


# company_info_path = os.path.join(os.path.dirname(__file__), 'company_info.csv')
# company_info_df = pd.read_csv(company_info_path)
# companies = [
#     {"id": idx + 1, "name": row["Company Name"], "symbol": row["Symbol"]}
#     for idx, row in company_info_df.iterrows()
# ]



# @app.route('/api/companies', methods=['GET'])
# def get_companies():
#     """Endpoint to fetch company data with current and predicted prices."""
#     result = []
#     for company in companies:
#         current_price, predicted_price = predict_stock(company['symbol'])
#         result.append({
#             "id": company["id"],
#             "name": company["name"],
#             "symbol": company["symbol"],
#             "current_price": round(current_price, 2) if current_price else None,
#             "predicted_price": round(predicted_price, 2) if predicted_price else None
#         })
#     return jsonify(result)


# @app.route('/company/<int:id>', methods=['GET'])
# def get_company(id):
#     """Endpoint to get company details with prediction."""
#     company = next((c for c in companies if c["id"] == id), None)
#     if not company:
#         return jsonify({"error": "Company not found"}), 404

#     current_price, predicted_price = predict_stock(company['symbol'])
#     if predicted_price is None:
#         return jsonify({"error": "Unable to predict price for this symbol"}), 400

#     return jsonify({
#         "name": company["name"],
#         "symbol": company["symbol"],
#         "predicted_price": round(predicted_price, 2) if predicted_price else None,
#         "current_price": round(current_price, 2) if current_price else None
#     })

# # @app.route('/api/update', methods=['POST'])
# # def update_price():
# #     """Endpoint to update a company's stock price."""
# #     data = request.json
# #     company_id = data.get("id")
# #     new_price = data.get("price")
    
# #     company = next((c for c in companies if c["id"] == company_id), None)
# #     if company and new_price:
# #         company["price"] = new_price
# #         return jsonify({"message": "Price updated successfully", "company": company})
# #     return jsonify({"error": "Invalid data"}), 400

# if __name__ == '__main__':
#     app.run(debug=True)

