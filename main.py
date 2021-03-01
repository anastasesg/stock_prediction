## Miscelaneous libraries
from tensorflow.keras import callbacks
from tqdm import tqdm
import numpy as np
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

## Libraries for the model creation
import tensorflow
import keras'
from keras.callbacks import ModelCheckpoint
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
def data_collection():
    data = yf.download(TICKER)
    data = data[[CATEGORY]]
    return data


def train_test_split(data, percent):
    scaled_data = SCALER(data)
    train_size = math.ceil(len(scaled_data) * percent)
    return scaled_data[:train_size], scaled_data[train_size:]


def feature_creation(data):
    x, y = [], []

    for i in range(WINDOW_SIZE, len(data)):
        x.append(data[i - WINDOW_SIZE:i, 0])
        y.append([data[i:i, 0]])
    
    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))

    return x, y


def build_model(train_x, train_y, test_x, test_y):
    MODEL.add(LSTM(300, return_sequences=True, input_shape=(train_x.shape[1], 1)))
    MODEL.add(LSTM(250, return_sequences=False))
    MODEL.add(Dense(1))
    MODEL.compile(optimizer='adam', loss='mean_squared_error')

    filename = os.path.join(PARENT_DIRECTORY, 'model_epoch_{epoch:02d}.hdf5')
    checkpoint = ModelCheckpoint(filepath=filename, monitor='val_loss', verbose=True, save_best_only=True, mode ='min')
    MODEL.fit(train_x, train_y, epochs = 10, batch_size = 100, callbacks = [checkpoint], validation_data = (test_x, test_y))
    

# Constants
TICKER = 'TSLA'
CATEGORY = 'Adj Close'
WINDOW_SIZE = 60
PARENT_DIRECTORY = 'saved_models'

SCALER = MinMaxScaler()
MODEL = Sequential()

# Data collection
data = data_collection()

# Data processing
train, test = train_test_split(data)
train_x, test_x = feature_creation(train)
train_y, test_y = feature_creation(test)

