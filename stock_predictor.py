import os
import mplfinance as mpf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from requests import models
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')


class StockPredictor:
    def __init__(self, stock):
        self.stockname = stock
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        if(not os.path.exists('./'+self.stockname.upper() + '_visualizations')):
            os.mkdir('./'+self.stockname.upper() + '_visualizations')

    def download_and_preprocess(self):
        '''
        downloads data of the given stock for the past 5 years and preprocesses it
        '''
        try:
            data = yf.download(self.stockname, period='5y')
        except:
            print('Please enter correct name of stock')

        data.dropna(inplace=True)
        data.drop('Adj Close', axis=1, inplace=True)
        self.data = data
        self.prices_yesterday = data.iloc[-1]
        scaled_data = self.scaler.fit_transform(data)
        x_train, y_train = [], []  # Data preprocessing
        for i in range(60, len(data)-60):
            x_train.append(scaled_data[i-60:i, 0:5])
            y_train.append(scaled_data[i, 0:5])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_test = scaled_data[len(data)-60:len(data), 0:5]
        self.training_set_shape = x_train.shape
        return x_train, y_train, x_test

    def build_lstm(self):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, LSTM
        self.model = Sequential()
        self.model.add(LSTM(units=50, return_sequences=True,
                            input_shape=(self.training_set_shape[1], self.training_set_shape[2])))
        self.model.add(LSTM(units=50))
        self.model.add(Dense(5))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def create_visualisations_and_save(self):
        mpf.plot(self.data.iloc[len(self.data)-60:len(self.data)], type='line', volume=True,
                 savefig='./'+self.stockname.upper() + '_visualizations/trend.png')

        mpf.plot(self.data.iloc[len(self.data)-60:len(self.data)], type='candle', volume=True,
                 savefig='./'+self.stockname.upper() + '_visualizations/candle_stick.png')

        mpf.plot(self.data.iloc[len(self.data)-60:len(self.data)], type='candle',
                 mav=(20), style='yahoo', volume=True, savefig='./'+self.stockname.upper() + '_visualizations/candle_stick_with_mav.png')
        print('Visualizations saved to device')

    def create_visualisations(self):
        print('Plotting current trend graph for requested stock...')
        mpf.plot(self.data.iloc[len(self.data) -
                                60:len(self.data)], type='line', volume=True, title='Line Plot')

        mpf.plot(self.data.iloc[len(self.data) -
                                60:len(self.data)], type='candle', volume=True, title='Candle-Stick Chart')

        mpf.plot(self.data.iloc[len(self.data)-60:len(self.data)], type='candle',
                 mav=(20), style='yahoo', volume=True, title='Candle-Stick Chart with Moving Average')

    def train_model(self, x_train, y_train):
        self.build_lstm()
        self.model.fit(x_train, y_train, epochs=5, batch_size=15, verbose=0)

    def predict_for_today(self, x_test):
        output = self.model.predict(x_test.reshape(
            1, x_test.shape[0], x_test.shape[1]))
        output = self.scaler.inverse_transform(output)
        self.prices_today = [round(output[0][0], 2), round(output[0][1], 2), round(
            output[0][2], 2), round(output[0][3], 2)]
        return(self.prices_today)

    def display(self, prices):
        print("Today's Price Predictions : ")
        print("Opening price Rs.", prices[0])
        print("High price Rs.", prices[1])
        print("Low price Rs.", prices[2])
        print("Closing price Rs.", prices[3])

    def recommend_stocks(self, stocks):
        max_profit = 0
        company_stock = 0
        company_prices = []
        for x in stocks:
            self.stockname = x
            x_train, y_train, x_test = self.download_and_preprocess()
            self.train_model(x_train, y_train)
            prices = self.predict_for_today(x_test)
            if(abs(prices[1]-prices[2]) > max_profit):
                max_profit = abs(prices[0]-prices[1])
                company_prices.append((np.mean(prices), x))
                company_stock = x
        company_prices.sort()
        print("This is a good time to buy the stocks of : ",
              company_prices[0][1])
        print("This is a good time to sell the stocks of : ",
              company_prices[-1][1])

        print("It is recommended to trade stocks of company for intraday purposes : ", company_stock)
        print("Max profit that can be earned is ",
              max_profit, " rupees per stock")
        return(company_stock, max_profit)
