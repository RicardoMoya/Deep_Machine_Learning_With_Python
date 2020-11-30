# -*- coding: utf-8 -*-
__author__ = 'RicardoMoya'

import pandas as pd

from keras.models import model_from_json

# Cargamos los datos
df = pd.read_csv("../data/calories_time_weight_speed.csv")
print(df.sample(5))

# Cambio de estructura de datos de entrada a  numpy
x_features = ['Tiempo', 'Peso', 'Velocidad']
X = df[x_features]
y = df['Calorias'].values

# leemos el Json y cargamos el modelo
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Cargamos los pesos
loaded_model.load_weights("best_weights.hdf5")
print("Modelo Cargado")

# evaluamos el modelo
loaded_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])
score = loaded_model.evaluate(X, y, verbose=2)
print("Loss: {:0.2f}".format(score[0]))
print("MSE: {:0.2f}".format(score[1]))
print("MAE: {:0.2f}".format(score[2]))
