# -*- coding: utf-8 -*-
__author__ = 'RicardoMoya'

import pandas as pd

from keras.models import Sequential
from keras.layers import Dense

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

# Compilamos el modelo
#   Función de perdida: Cross Entropy. Al ser un problema de clasificación binaria: binary_crossentropy
#   Optimizador: ADAM
#   Métricas a monitorizar: Accuracy
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Ajuste del Modelo
#   Epochs = 100
#   Batch Size = 8
model.fit(X, y, epochs=100, batch_size=8, verbose=2)

# Evaluación del modelo con los mismos datos de entrenamiento
scores = model.evaluate(X, y)
print("\nNombres de las métricas: {}".format(model.metrics_names))
print("Resultados de las métricas: {}".format(scores))
for index, metric in enumerate(model.metrics_names):
    print("{} : {}".format(metric, scores[index]))

# Predicción de resultados
X_predict = X[0:3]
y_real = y[0:3]
print("\nResultados Reales")
for i in range(len(X_predict)):
    print("elem {} - Clase {} - X: {}".format((i + 1), y_real[i], X_predict[i]))

# Salida de la Red Neuronal
print("\nSalida de la Red Neuronal: Clase 0 < 0.5, Clase 1 >= 0.5")
y_output = model.predict(X_predict)
for i in range(len(y_output)):
    print("elem {} - Salida: {:0.2f} - Clase predicha: {} - Clase Real: {} - Acierto: {}"
          .format((i + 1),
                  y_output[i][0],
                  (0 if y_output[i][0] < 0.5 else 1),
                  y_real[i],
                  y_real[i] == (0 if y_output[i][0] < 0.5 else 1)))

# Predicción de la Red Neuronal
print("\nPredicción de la Red Neuronal")
y_output_class = model.predict_classes(X_predict)
for i in range(len(y_output_class)):
    print("elem {} - Salida: {:0.2f} - Clase Real: {} - Acierto: {}"
          .format((i + 1),
                  y_output_class[i][0],
                  y_real[i],
                  y_real[i] == y_output_class[i][0]))
