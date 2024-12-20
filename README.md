#  Long Short-Term Memory Recurrent Neural Network for prediction of Nifty50 index from historical data

## Stock Price Prediction Using LSTM

This project aims to predict the next day's stock prices based on historical data from the Nifty 50 index using Long Short-Term Memory (LSTM) neural networks. The model is built using TensorFlow and Keras to predict stock prices, leveraging time-series data for training and testing. This app is deployed on streamlit platform and can be used for testing or predicting the Nifty50 future open, close, %change in price.

## Overview

The goal of this project is to use machine learning techniques, specifically LSTM (a type of Recurrent Neural Network), to predict stock prices for the Nifty 50 index. The model uses historical stock data to make predictions on future stock prices, which can be useful for investors and analysts.

## Data Preprocessing
The data used in this project is from the Nifty 50 index. The preprocessing steps include:

1. Downloading the historical stock price data.
2. Normalizing the data using MinMaxScaler.
3. Splitting the data into training and testing sets.
4. Model Architecture
   * The LSTM model is built with the following architecture:
   * Input Layer: Takes in the preprocessed stock price data.
   * LSTM Layer: The core of the model, where the sequential data is processed.
   * Dense Layer: A fully connected layer that outputs the predicted stock price.
5. Training the Model
6. Once the data is preprocessed, the LSTM model is trained using the training dataset. The model is then evaluated using the testing dataset.

## Making Predictions
After training, the model can be used to make predictions on the next day's stock price.

## Example usage of the trained model for prediction
1. predicted_price = model.predict(input_data)
2. Streamlit app for prediction
   
## Results
The performance of the model can be evaluated using metrics such as Mean Absolute Error (MAE) or Root Mean Squared Error (RMSE). A graph of the predicted vs. actual stock prices can be plotted to visualize the model's performance.


