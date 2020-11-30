# -*- coding: utf-8 -*-
__author__ = 'RicardoMoya'

import pandas as pd

from keras.layers import Dense, Dropout
from keras.models import Sequential

# Cargamos los datos
df = pd.read_csv("../data/diabetes.csv")
print(df.sample(5))

# Cambio de Estructura de datos a Numpy
x_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
              'DiabetesPedigreeFunction', 'Age']
X = df[x_features].values
y = df['Outcome'].values

# Definimos la Arquitectura de la Red Neuronal
model = Sequential()
model.add(Dropout(0.2, input_shape=(8,)))   # Dropout capa de entrada con 8 Neurona
model.add(Dense(12, activation='relu'))     # Capa oculta 1 con 12 Neuronas y función de activación RELU
model.add(Dropout(0.2))                     # Dropout Capa oculta 1
model.add(Dense(8, activation='relu'))      # Capa oculta 2 con 8 Neuronas y función de activación RELU
model.add(Dropout(0.2))                     # Dropout Capa oculta 2
model.add(Dense(1, activation='sigmoid'))   # Capa de Salida de 1 Neurona con función de activación SIGMOID

# Imprimimos por pantalla la arquitectura de la red definida
print(model.summary())

# Compilamos el modelo
#   Función de perdida: Cross Entropy. Al ser un problema de clasificación binaria: binary_crossentropy
#   Optimizador: ADAM
#   Métricas a monitorizar: Accuracy, Precision, Recall y F1
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Ajuste del Modelo
#   Epochs = 100
#   Batch Size = 8
model.fit(X, y, validation_split=0.2, epochs=100, batch_size=8, verbose=2)

# Evaluación del modelo con los mismos datos de entrenamiento
metrics = model.evaluate(X, y)
print("\nNombres de las métricas: {}".format(model.metrics_names))
print("Resultados de las métricas: {}".format(metrics))
for index, metric in enumerate(model.metrics_names):
    print("{} : {}".format(metric, metrics[index]))
