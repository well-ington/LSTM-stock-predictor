# -*- coding: utf-8 -*-

import requests
from sklearn import preprocessing
import numpy as np
from keras.models import Model, Sequential
import keras.layers as k
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

URL = 'https://api.coindesk.com/v1/bpi/historical/close.json?start=2014-07-18&end=2017-02-22'
URL_TEST = 'https://api.coindesk.com/v1/bpi/historical/close.json?start=2017-02-22&end=2018-02-22'
r = requests.get(url = URL)
test = requests.get(url = URL_TEST)
data_t = test.json()
t = np.array(list(data_t['bpi'].values()))

data = r.json()
training_rnn = data['bpi']

stock_list = np.array(list(training_rnn.values()))
x_training_scaled = preprocessing.scale(stock_list)

sc = preprocessing.MinMaxScaler(feature_range=(0,1))
x_transformed = stock_list.reshape(-1,1)
x_min_max = sc.fit_transform(x_transformed)
"""plt.plot(x_training_scaled)"""

X_train = []
Y_train = []

for i in range(60, 770):
    X_train.append(x_min_max[i-60:i, 0])
    Y_train.append(x_min_max[i, 0])

X_train, Y_train = np.array(X_train), np.array(Y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


reg = Sequential()

reg.add(k.LSTM(300, return_sequences = True, input_shape = (X_train.shape[1], 1)))
reg.add(k.Dropout(0.2))

reg.add(k.LSTM(300, return_sequences = True))
reg.add(k.Dropout(0.2))

reg.add(k.LSTM(150, return_sequences = True))
reg.add(k.Dropout(0.2))

reg.add(k.LSTM(75))
reg.add(k.Dropout(0.2))

reg.add(k.Dense(1, activation='sigmoid'))

reg.compile(optimizer = 'adam', loss = 'mean_squared_error')

reg.fit(X_train, Y_train, epochs = 18, batch_size = 32)


#getting reference to 2020
t = t.reshape(-1,1)
t = sc.fit_transform(t)

generated = x_min_max[len(x_min_max)-61:len(x_min_max)]

x_tester = []

x_generated = []
x_predicted = []

for i in range (0, 360):
    x_batch = []
    a = x_min_max[len(x_min_max)-60:len(x_min_max)]
    togo = np.append(a, x_predicted)
    x_batch = togo[len(togo) - 60:len(togo)]
    x_batch = x_batch.reshape(1,-1)
    x_batch = np.array(x_batch)
    x_batch = np.reshape(x_batch, (x_batch.shape[0], x_batch.shape[1], 1))
    tempor = reg.predict(x_batch)
    x_predicted = np.append(x_predicted, tempor)

plt.plot(x_predicted)
dolarprice = sc.inverse_transform(x_predicted.reshape(-1,1))
plt.plot(dolarprice)
plt.plot(t)

for i in range(60, 914):
    x_tester.append(t[i-60:i,0])

x_tester = np.array(x_tester)
x_tester = np.reshape(x_tester, (x_tester.shape[0], x_tester.shape[1], 1))

predicted_stock = reg.predict(x_tester)
plt.plot(predicted_stock)
plt.plot(t)



