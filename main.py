## Miscelaneous libraries
from tqdm import tqdm
import numpy as np
import math
import os

## Libraries for the model creation
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import load_model

## Libraries for data collection, presentation and processing
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# Constants
TICKER = 'TSLA'
CATEGORY = 'Adj Close'
TRAIN_PERCENT = 0.9
WINDOW_SIZE = 60

SCALER = MinMaxScaler()
MODEL = Sequential()

# Data collection
data = yf.download(TICKER)
data = data[[CATEGORY]]

# Data processing
train_size = math.ceil(len(data) * TRAIN_PERCENT)
train, test = data[:train_size], data[train_size:]

# Feature creation
train_x, test_x, train_y, test_y = [], [], [], []
for i in range(WINDOW_SIZE, len(train)):
    train_x.append(data[i - WINDOW_SIZE:i, 0])
    train_y.append([data[i:i, 0]])

for i in range(WINDOW_SIZE, len(test)):
    test_x.append(data[i - WINDOW_SIZE:i, 0])
    test_y.append([data[i:i, 0]])

train_x, train_y = np.array(train_x), np.array(train_y)
test_x, test_y = np.array(test_x), np.array(test_y)

train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

