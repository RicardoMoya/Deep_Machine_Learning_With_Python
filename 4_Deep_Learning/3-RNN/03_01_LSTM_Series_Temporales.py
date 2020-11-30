# -*- coding: utf-8 -*-
__author__ = 'RicardoMoya'

import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error


# Función para construir las series temporales y la salida
# Ejm: [1, 2, 3, 4, 5, 6, 7, 8, 9]
# mirar_atras = 3
# X = [[1, 2, 3], [2, 3, 4], [3, 4, 5], ...]
# y = [4, 5, 6, ...]
def create_dataset(dataset, mirar_atras=1):
    data_x, data_y = [], []
    for i in range(len(dataset) - mirar_atras - 1):
        a = dataset[i:(i + mirar_atras), 0]
        data_x.append(a)
        data_y.append(dataset[i + mirar_atras, 0])
    return np.array(data_x), np.array(data_y)


# Cargamos los datos
df = read_csv('../data/airline-passengers.csv', usecols=[1], engine='python')
dataset = df.values
dataset = dataset.astype('float32')

# Normalizamos los datos
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Datos de entrenamiento y test - Dividimos el dataset temporalmente
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

# Creamos el dataset marcando la ventana temporal
MIRAR_ATRAS = 4
train_x, train_y = create_dataset(train, MIRAR_ATRAS)
test_x, test_y = create_dataset(test, MIRAR_ATRAS)
# reshape: [samples, time steps, features]
train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

# Definimos la Arquitectura de la Red Neuronal
#   Entrada de 1 elementos de "mirar_atras" posiciones
#   Capa LSTM de 4 módulos de memoria
#   Capa de Salida de 1 Neurona con función de activación lineal
model = Sequential()
model.add(LSTM(4, input_shape=(1, MIRAR_ATRAS)))
model.add(Dense(1, activation='linear'))

# Imprimimos por pantalla la arquitectura de la red definida
print(model.summary())

# Compilamos el modelo
#   Función de perdida:MSE
#   Optimizador: ADAM
#   Métricas a monitorizar: función de perdida = MSE
model.compile(loss='mean_squared_error', optimizer='adam')

# Ajuste del Modelo
#   Epochs = 100
#   Batch Size = 1
model.fit(train_x, train_y, epochs=100, batch_size=1, verbose=2)

# Realizamos las predicciones
train_predict = model.predict(train_x)
test_predict = model.predict(test_x)

# "Desnormalizamos" los datos
train_predict = scaler.inverse_transform(train_predict)
train_y = scaler.inverse_transform([train_y])
test_predict = scaler.inverse_transform(test_predict)
test_y = scaler.inverse_transform([test_y])

# Evaluación del modelo (MAE) con los datos de entrenamiento y test
mse_train = mean_absolute_error(y_true=train_y[0], y_pred=train_predict[:, 0])
print('MSE Train = {:0.2f}'.format(mse_train))
mse_test = mean_absolute_error(y_true=test_y[0], y_pred=test_predict[:, 0])
print('MSE Test = {:0.2f}'.format(mse_test))

# Pintamos los resultados
train_predict_plot = np.empty_like(dataset)
train_predict_plot[:, :] = np.nan
train_predict_plot[MIRAR_ATRAS:len(train_predict) + MIRAR_ATRAS, :] = train_predict

test_predict_plot = np.empty_like(dataset)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (MIRAR_ATRAS * 2) + 1:len(dataset) - 1, :] = test_predict

plt.plot(scaler.inverse_transform(dataset))
plt.plot(train_predict_plot)
plt.plot(test_predict_plot)
plt.show()
