# -*- coding: utf-8 -*-
__author__ = 'RicardoMoya'

import pandas as pd

from keras.layers import Dense
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

# Cargamos los datos
df = pd.read_csv("../data/iris.csv")
print(df.sample(5))

# Cambio de estructura de datos de entrada a  numpy
x_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = df[x_features].values

# Transformamos la clase de las flores a un array codificado de 3 clases:
# [clase_1, clase_2, clase_3] ya que la red tendrá como salida 3 neuronas
y_labels = df['class'].values
y_encode = LabelEncoder().fit_transform(y_labels)
y = np_utils.to_categorical(y_encode)

# Definimos la Arquitectura de la Red Neuronal
# Creamos el modelo
model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Imprimimos por pantalla la arquitectura de la red definida
print(model.summary())

# Compilamos el modelo
#   Función de perdida: Cross Entropy. Al ser un problema de clasificación múltiple: categorical_crossentropy
#   Optimizador: ADAM
#   Métricas a monitorizar: Accuracy
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Ajuste del Modelo
#   Epochs = 100
#   Batch Size = 4
model.fit(X, y, epochs=100, batch_size=4, verbose=2)

# Evaluación del modelo con los mismos datos de entrenamiento
metrics = model.evaluate(X, y)
print("\nNombres de las métricas: {}".format(model.metrics_names))
print("Resultados de las métricas: {}".format(metrics))
for index, metric in enumerate(model.metrics_names):
    print("{} : {}".format(metric, metrics[index]))

# Predicción de resultados
X_predict = X[0:3]
y_real = y[0:3]
y_real_encode = y_encode[0:3]
print("\nResultados Reales")
for i in range(len(X_predict)):
    print("elem {} - Clase {} - X: {}".format((i + 1), y_real[i], X_predict[i]))

# Predicción de la Red Neuronal
print("\nSalida y Predicción de la Red Neuronal")
y_output = model.predict(X_predict)
y_output_class = model.predict_classes(X_predict)
for i in range(len(y_output_class)):
    print("elem {} - Salida: {} - Predicción: {} - Clase Real: {} -> {} - Acierto: {}"
          .format((i + 1),
                  y_output[i],
                  y_output_class[i],
                  y_real_encode[i],
                  y_real[i],
                  y_output_class[i] == y_real_encode[i]))
