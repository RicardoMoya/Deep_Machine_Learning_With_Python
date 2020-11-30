# -*- coding: utf-8 -*-
__author__ = 'RicardoMoya'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
#   Epochs = 200
#   Batch Size = 8
history = model.fit(X, y, validation_split=0.2, epochs=200, batch_size=8, verbose=2)


# Métricas por Epoch
def plot_metric(history, name):
    metric_train = np.array(history.history[name])
    metric_test = np.array(history.history['val_{}'.format(name)])
    acum_avg_metric_train = np.cumsum(metric_train) / (np.arange(metric_train.shape[-1]) + 1)
    acum_avg_metric_test = np.cumsum(metric_test) / (np.arange(metric_test.shape[-1]) + 1)
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title('{} - Epochs'.format(name))
    plt.plot(metric_train, label='{} Train'.format(name))
    plt.plot(metric_test, label='{} Test'.format(name))
    plt.legend(loc='upper center')
    plt.subplot(1, 2, 2)
    plt.title('AVG ACCUMULATIVE {} - Epochs'.format(name))
    plt.plot(acum_avg_metric_train, label='{} Train'.format(name))
    plt.plot(acum_avg_metric_test, label='{} Test'.format(name))
    plt.legend(loc='upper center')
    plt.show()


# Función de perdida
plot_metric(history=history, name='loss')

# Accuracy
plot_metric(history=history, name='accuracy')

# Precision
plot_metric(history=history, name='precision_k')

# Recall
plot_metric(history=history, name='recall_k')

# F1
plot_metric(history=history, name='f1_k')
