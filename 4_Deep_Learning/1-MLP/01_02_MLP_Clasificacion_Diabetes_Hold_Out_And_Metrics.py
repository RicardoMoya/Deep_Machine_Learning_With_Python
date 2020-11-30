# -*- coding: utf-8 -*-
__author__ = 'RicardoMoya'

import pandas as pd

from keras import backend as K
from keras.layers import Dense
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
#   Capa de entrada de 8 Neuronas
#   Capa oculta 1 con 12 Neuronas y función de activación RELU
#   Capa oculta 2 con 8 Neuronas y función de activación RELU
#   Capa de Salida de 1 Neurona con función de activación SIGMOID (Valores de salida entre {0-1})
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Imprimimos por pantalla la arquitectura de la red definida
print(model.summary())


# Implementamos las métricas del Precision, Recall y F1
def precision_k(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall_k(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1_k(y_true, y_pred):
    precision = precision_k(y_true, y_pred)
    recall = recall_k(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# Compilamos el modelo
#   Función de perdida: Cross Entropy. Al ser un problema de clasificación binaria: binary_crossentropy
#   Optimizador: ADAM
#   Métricas a monitorizar: Accuracy, Precision, Recall y F1
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', precision_k, recall_k, f1_k])

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
