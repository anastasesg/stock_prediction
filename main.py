import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date, datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint

import getopt, sys

tf.random.set_seed(7)
np.random.seed(7)


class Predictor:
    def __init__(self, percent, window_size, ticker, category, par_dir, score_name):
        self.__window_size = window_size
        self.ticker = ticker
        self.category = category
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model_dir = par_dir
        self.model = Sequential()
        self.score = score_name
        self.load_data(percent)
        self.process_data()

    def create_features(self, data):
        x, y = [], []
        for i in range(self.__window_size, len(data)):
            x.append(data[i - self.__window_size:i, 0])
            y.append([data[i:i, 0]])
        return np.array(x), np.array(y)

    def load_data(self, percent):
        print('Loading Data...')
        self.data = yf.download(self.ticker)
        self.data = self.data[[self.category]]
        self.train_size = math.ceil(len(self.data) * percent)
        self.train, self.valid = self.data[:self.train_size], self.data[self.train_size:]

    def process_data(self):
        dataset = self.data.values
        dataset = dataset.astype('float32')

        scaled_data = self.scaler.fit_transform(dataset)

        train, test = scaled_data[:self.train_size, :], scaled_data[self.train_size - self.__window_size:, :]

        self.x_train, self.y_train = self.create_features(train)
        self.x_test, self.y_test = self.create_features(test)

        self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], self.x_train.shape[1], 1))
        self.x_test = np.reshape(self.x_test, (self.x_test.shape[0], self.x_test.shape[1], 1))

    def plot_data(self):
        plt.figure()
        plt.title('Model for {}'.format(self.ticker))
        plt.xlabel('Date')
        plt.ylabel('{} Price USD ($)'.format(self.category))
        plt.plot(self.train[self.category])
        plt.plot(self.valid[[self.category, 'Predictions']])
        plt.legend(['Train', 'Valid', 'Predictions'], loc='lower right')
        plt.show()

    def build_model(self, optimizer, loss, save, epochs, batch_size):
        self.model.add(LSTM(300, return_sequences=True, input_shape=(self.x_train.shape[1], 1)))
        self.model.add(LSTM(250, return_sequences=True))
        self.model.add(LSTM(200, return_sequences=False))
        self.model.add(Dense(50))
        self.model.add(Dense(1))
        self.model.compile(optimizer=optimizer, loss=loss)

        if save:
            filepath = os.path.join(self.model_dir, 'model_epoch_{epoch:02d}.hdf5')
            checkpoint = ModelCheckpoint(
                filepath=filepath,
                monitor='val_loss',
                verbose=True,
                save_best_only=True,
                mode ='min'
            )
            self.model.fit(
                self.x_train,
                self.y_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[checkpoint],
                validation_data=(self.x_test, self.y_test)
            )
        else:
            self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size)

    def model_selection(self):
        score_list, model_list = [], []
        filelength = len(os.listdir(self.model_dir))
        for num, file in enumerate(os.listdir(self.model_dir)):
            filepath = os.path.join(self.model_dir, file)
            print('Checking file {0}/{1} : {2}'.format(num + 1, filelength, file), end="\r", flush=True)
            
            self.model = load_model(filepath)
            predict = self.scaler.inverse_transform(self.model.predict(self.x_test))

            model_list.append(file)
            score_list.append(self.score(self.y_test, predict))
        print()
        score_list, model_list = zip(*sorted(zip(score_list, model_list), reverse=True))
        return os.path.join(self.model_dir, model_list[0])

    def get_model(self, optimizer='adam', loss='mean_squared_error', save=False, epochs=10, batch_size=10):
        if len(os.listdir(self.model_dir)) == 0 or save:
            self.build_model(optimizer, loss, save, epochs, batch_size)
        # model_file = self.model_selection()
        # print(model_file)
        self.model = load_model(os.path.join(self.model_dir, os.listdir(self.model_dir)[-1]))
    
    def predict_tomorrow(self):
        last_window = np.array([self.scaler.transform(self.data[-self.__window_size:].values)])
        last_window = np.reshape(last_window, (last_window.shape[0], last_window.shape[1], 1))

        return self.scaler.inverse_transform(self.model.predict(last_window))


if __name__ == '__main__':
    predictor = Predictor(
        percent=0.95,
        window_size=30,
        ticker='HLTOY',
        category='Adj Close',
        par_dir='saved_models',
        score_name=mean_squared_error
    )
    predictor.get_model(
        save=False,
        # epochs=1,
        # batch_size=30
    )
    prediction = predictor.scaler.inverse_transform(predictor.model.predict(predictor.x_test))
    predictor.valid['Predictions'] = prediction
    predictor.plot_data()
    print(predictor.valid)
    print(predictor.predict_tomorrow())