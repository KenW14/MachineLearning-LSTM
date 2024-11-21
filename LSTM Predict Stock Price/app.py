from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

import yfinance as yf
import numpy as np
import pandas as pd
import base64
import io
import matplotlib.pyplot as plt
import matplotlib

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from tensorflow.keras.optimizers import Adam
from scipy.stats import pearsonr

matplotlib.use('Agg')

app = Flask(__name__)

def load_custom_model(model_path):
    return load_model(model_path, custom_objects={'mse': 'mse'})

model = load_custom_model('lstm_model.h5')
scaler = MinMaxScaler()
n_steps = 30
days_to_predict = 30
optimizer = Adam() 
model.compile(optimizer=optimizer, loss='mse')

def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        end_ix = i + n_steps
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def plot_loss(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    return plt

def plot_training_and_testing_data(stock_data, y_train_inv, y_train_pred_inv, y_test_inv, y_test_pred_inv, train_size):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(stock_data.index[n_steps:train_size + n_steps], y_train_inv, label='Actual (Train)', color='blue')
    plt.plot(stock_data.index[n_steps:train_size + n_steps], y_train_pred_inv, label='Predicted (Train)', color='green')
    plt.title('Prediksi Harga Saham (BBCA) - Data Pelatihan')
    plt.xlabel('Tanggal')
    plt.ylabel('Harga Saham')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(stock_data.index[train_size + n_steps:], y_test_inv, label='Actual (Test)', color='red')
    plt.plot(stock_data.index[train_size + n_steps:], y_test_pred_inv, label='Predicted (Test)', color='green')
    plt.title('Prediksi Harga Saham (BBCA) - Data Pengujian')
    plt.xlabel('Tanggal')
    plt.ylabel('Harga Saham')
    plt.legend()
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    return plot_base64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_stock_data', methods=['GET'])
def get_stock_data():
    symbol = request.args.get('symbol')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    try:
        end_date_adjusted = (datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        stock_data = yf.download(symbol, start=start_date, end=end_date_adjusted)
        adj_close_data = stock_data['Adj Close']

        stock_data_first = stock_data.head(5)
        stock_data_last = stock_data.tail(5)
        stock_data_combined = pd.concat([stock_data_first, stock_data_last])
        stock_data_combined.index = stock_data_combined.index.strftime('%Y-%m-%d')
        stock_data_dict = stock_data_combined.to_dict(orient='index')
        
        # Hanya tampilkan grafik jika endpoint ini yang dipanggil
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(stock_data.index, stock_data['Adj Close'], label='Adj Close')
        ax.set_xlabel('Tanggal')
        ax.set_ylabel('Adj Close')
        ax.set_title(f'Grafik {symbol}')
        ax.legend()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return jsonify({'data': stock_data_dict, 'plot': image_base64, 'error': None})
    except Exception as e:
        return jsonify({'data': None, 'plot': None, 'error': str(e)})

@app.route('/predict_stock_price', methods=['GET'])
def predict_stock_price():
    symbol = request.args.get('symbol')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    period = int(request.args.get('period', 30))  # Default to 30 days
    
    try:
        end_date_adjusted = (datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        stock_data = yf.download(symbol, start=start_date, end=end_date_adjusted)

        if stock_data.empty:
            return jsonify({'predictions': None, 'dates': None, 'error': 'No data available for the specified symbol and date range.'})

        adj_close_data = stock_data['Adj Close']
        scaler.fit(np.array(adj_close_data).reshape(-1, 1))
        adj_close_normalized = scaler.transform(np.array(adj_close_data).reshape(-1, 1))
        X, y = prepare_data(adj_close_normalized, n_steps)
        train_size = int(len(X) * 0.8)
        X_test = X[train_size:]
        X_future = np.reshape(X_test[-1], (1, n_steps, 1))
        
        future_predictions = []
        future_dates = []

        for i in range(period):
            prediction = model.predict(X_future)
            future_predictions.append(prediction.flatten()[0])
            next_date = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=i+1)
            future_dates.append(next_date.strftime('%Y-%m-%d'))
            X_future = np.roll(X_future, -1)
            X_future[0, -1, 0] = prediction.flatten()[0]
        
        future_predictions_inv = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        return jsonify({'predictions': future_predictions_inv.flatten().tolist(), 'dates': future_dates, 'error': None})
    except Exception as e:
        return jsonify({'predictions': None, 'dates': None, 'error': str(e)})

@app.route('/evaluate_model', methods=['GET'])
def evaluate_model():
    symbol = request.args.get('symbol')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    try:
        end_date_adjusted = (datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        stock_data = yf.download(symbol, start=start_date, end=end_date_adjusted)

        if stock_data.empty:
            return jsonify({'plot': None, 'loss_plot': None, 'error': f'Tidak ada data tersedia untuk simbol {symbol} dalam rentang tanggal yang ditentukan.'})

        adj_close_data = stock_data['Adj Close']
        scaler.fit(np.array(adj_close_data).reshape(-1, 1))
        adj_close_normalized = scaler.transform(np.array(adj_close_data).reshape(-1, 1))
        X, y = prepare_data(adj_close_normalized, n_steps)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Evaluasi model
        history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test), verbose=1)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_train_pred_inv = scaler.inverse_transform(y_train_pred)
        y_test_pred_inv = scaler.inverse_transform(y_test_pred)
        y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Menghitung metrik evaluasi
        train_rmse = np.sqrt(mean_squared_error(y_train_inv, y_train_pred_inv))
        train_mape = np.mean(np.abs((y_train_inv - y_train_pred_inv) / y_train_inv)) * 100
        train_pearson = pearsonr(y_train_inv.flatten(), y_train_pred_inv.flatten())[0]

        test_rmse = np.sqrt(mean_squared_error(y_test_inv, y_test_pred_inv))
        test_mape = np.mean(np.abs((y_test_inv - y_test_pred_inv) / y_test_inv)) * 100
        test_pearson = pearsonr(y_test_inv.flatten(), y_test_pred_inv.flatten())[0]

        plot = plot_training_and_testing_data(stock_data, y_train_inv, y_train_pred_inv, y_test_inv, y_test_pred_inv, train_size)
        loss_plot = plot_loss(history)

        # Mengonversi plot menjadi format base64 untuk dikirimkan sebagai respons JSON
        buffer = io.BytesIO()
        loss_plot.savefig(buffer, format='png')
        buffer.seek(0)
        loss_plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return jsonify({
            'plot': plot,
            'loss_plot': loss_plot_base64,
            'train_rmse': train_rmse,
            'train_mape': train_mape,
            'train_pearson': train_pearson,
            'test_rmse': test_rmse,
            'test_mape': test_mape,
            'test_pearson': test_pearson,
            'error': None
        })
    except Exception as e:
        return jsonify({'plot': None, 'loss_plot': None, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)