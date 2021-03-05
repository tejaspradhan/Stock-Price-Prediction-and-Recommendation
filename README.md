# Stock-Price-Prediction-and-Recommendation

Stock price prediction and recommendation is a python application which predicts the next day's open,high,low and closing values based on past data. Along with prediction, the application can also be used as a recommendation engine for the best performing stocks among the ones given as input. This functionality has also been implemented as a python library and can be imported as a class file(stock_predictor.py) for further usage and extension.

# Project Description
This application uses a time series linear LSTM model to predict stock prices. Given a particular stock, the application automatically fetches the pasr stock price data for that stock from the yahoo finance API and trains the LSTM regression model in real-time. Based on this it makes a prediction for the next day.
