# -*- coding: utf-8 -*-
__author__ = 'RicardoMoya'

import pandas as pd

from keras.callbacks import ModelCheckpoint
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

# Compilamos el modelo
#   Función de perdida: Error Cuadrático Medio: mean_squared_error
#   Optimizador: adam
#   Métricas a monitorizar: MSE y MAE
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])

# Guardamos la arquitectura de la red en un JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Guardamos los pesos de la red en un fichero hdf5
file_weights = "best_weights.hdf5"
checkpoint = ModelCheckpoint(file_weights,
                             monitor='mae',
                             save_best_only=True,
                             mode='min',
                             verbose=2)
callbacks_list = [checkpoint]

# Ajuste del Modelo
#   Epochs = 200
#   Batch Size = 32
model.fit(X, y, epochs=200, batch_size=32, callbacks=callbacks_list, verbose=2)
