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


# Define functions
def train_test_split(data, percent):
    train_size = math.ceil(len(data) * percent)
    return data[:train_size], data[train_size:]

def feature_creation(data):
    x, y = [], []

    for i in range(WINDOW_SIZE, len(data)):
        x.append(data[i - WINDOW_SIZE:i, 0])
        y.append([data[i:i, 0]])
    
    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))

    return x, y

# Constants
TICKER = 'TSLA'
CATEGORY = 'Adj Close'
WINDOW_SIZE = 60

SCALER = MinMaxScaler()
MODEL = Sequential()

# Data collection
data = yf.download(TICKER)
data = data[[CATEGORY]]

# Data processing
train, test = train_test_split(data)

# Feature creation
train_x, test_x = feature_creation(train)
train_y, test_y = feature_creation(test)

