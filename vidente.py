import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

import os
import random
import pickle

def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:i+time_step, 0])
        y.append(dataset[i+time_step, 0])
    return np.array(X), np.array(y)

# seed
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# baixar os dados
ticker = "PETR4.SA"
data = yf.download(ticker, start="2015-01-01", end="2024-01-01")
# Utilizar apenas o preço de fechamento
close_prices = data[['Close']]

# normalizaçao
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)
#with open('scaler.pkl', 'wb') as f:
#    pickle.dump(scaler, f)

#criar sequencias temporais
time_step = 60
X, y = create_dataset(scaled_data, time_step)

# cria dataset de treino e teste
train_size = int(len(X) * 0.8)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Ajustar formato para LSTM [amostras, timesteps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# cria modelo LSTM
model = Sequential()
model.add(Input(shape=(time_step, 1)))

model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(
    optimizer='adam',
    loss='mean_squared_error'
    )

# treinar modelo
early_stop = EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
    )

# save model
# model.save("b3_lstm_model.keras")

# previsoes
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Desnormalizar os dados
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

y_train_real = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

# avaliação do modelo
rmse = np.sqrt(mean_squared_error(y_test_real, test_predict))
print(f"RMSE do modelo: {rmse:.2f}")

plt.figure(figsize=(14,6))
plt.plot(y_test_real, label="Preço Real", color='blue')
plt.plot(test_predict, label="Previsão LSTM", color='red')
plt.title("Previsão de Fechamento - PETR4 (Multivariado)")
plt.xlabel("Tempo")
plt.ylabel("Preço (R$)")
plt.legend()
plt.show()