## Miscelaneous libraries
from tqdm import tqdm
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

SCALER = MinMaxScaler()
MODEL = Sequential()

# Data collection
data = yf.download(TICKER)
data = data[[CATEGORY]]

train_size = math.ceil(len(data) * TRAIN_PERCENT)
train, valid = data[:train_size], data[train_size:]

