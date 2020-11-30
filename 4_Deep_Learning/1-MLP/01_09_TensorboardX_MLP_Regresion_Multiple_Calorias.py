# -*- coding: utf-8 -*-
__author__ = 'RicardoMoya'

import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.callbacks import TensorBoard
from keras.layers import Dense
from keras.models import Sequential


# Cargamos los datos
df = pd.read_csv("../data/calories_time_weight_speed.csv")
print(df.sample(5))

# Cambio de estructura de datos de entrada a  numpy
x_features = ['Tiempo', 'Peso', 'Velocidad']
X = df[x_features]
y = df['Calorias'].values

# Definimos la Arquitectura de la Red Neuronal
# Creamos el modelo
model = Sequential()
model.add(Dense(12, input_dim=3, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

# Imprimimos por pantalla la arquitectura de la red definida
print(model.summary())

# Compilamos el modelo
#   Función de perdida: Error Cuadrático Medio: mean_squared_error
#   Optimizador: adam
#   Métricas a monitorizar: MSE y MAE
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])

# Ajuste del Modelo
#   Epochs = 200
#   Batch Size = 32
now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs('tensorboard_logs/{}'.format(now))
path = os.path.join('tensorboard_logs', now)
history = model.fit(X, y, validation_split=0.2, epochs=200, batch_size=32, verbose=2,
                    callbacks=[TensorBoard(log_dir=path,
                                           histogram_freq=2,
                                           write_graph=True,
                                           write_images=True)])

# Instanciar el Tensorboard: >> tensorboard --logdir ./tensorboard_logs/
# Abrir en el navegador: http://localhost:6006/

# Evaluación del modelo con los mismos datos de entrenamiento
metrics = model.evaluate(X, y)
print("\nNombres de las métricas: {}".format(model.metrics_names))
print("Resultados de las métricas: {}".format(metrics))
for index, metric in enumerate(model.metrics_names):
    print("{} : {}".format(metric, metrics[index]))

# Pintamos las métricas por epoch
def plot_metric(history, name, remove_first=0):
    metric_train = np.array(history.history[name])[remove_first:]
    metric_test = np.array(history.history['val_{}'.format(name)])[remove_first:]
    acum_avg_metric_train = (np.cumsum(metric_train) / (np.arange(metric_train.shape[-1]) + 1))[remove_first:]
    acum_avg_metric_test = (np.cumsum(metric_test) / (np.arange(metric_test.shape[-1]) + 1))[remove_first:]
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title('{} - Epochs'.format(name))
    plt.plot(metric_train, label='{} Train'.format(name))
    plt.plot(metric_test, label='{} Test'.format(name))
    plt.grid()
    plt.legend(loc='upper center')
    plt.subplot(1, 2, 2)
    plt.title('AVG ACCUMULATIVE {} - Epochs'.format(name))
    plt.plot(acum_avg_metric_train, label='{} Train'.format(name))
    plt.plot(acum_avg_metric_test, label='{} Test'.format(name))
    plt.grid()
    plt.legend(loc='upper center')
    plt.show()

# Función de perdida
plot_metric(history=history, name='loss')

# MSE
plot_metric(history=history, name='mse')

# MAE
plot_metric(history=history, name='mae')
