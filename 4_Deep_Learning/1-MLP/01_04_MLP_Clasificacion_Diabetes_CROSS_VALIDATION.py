# -*- coding: utf-8 -*-
__author__ = 'RicardoMoya'

import numpy as np
import pandas as pd

from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import KFold

# Cargamos los datos
df = pd.read_csv("../data/diabetes.csv")
print(df.sample(5))

# Cambio de Estructura de datos a Numpy
x_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
              'DiabetesPedigreeFunction', 'Age']
X = df[x_features].values
y = df['Outcome'].values

# Definimos el Cross Validation en 10 grupos
k_fold = KFold(n_splits=10, random_state=0, shuffle=True)
folds_train = list()
folds_test = list()
for train, test in k_fold.split(X, y):
    # Definimos la Arquitectura de la Red Neuronal
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compilamos el modelo
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Ajuste del Modelo
    model.fit(X[train], y[train], validation_split=0.2, epochs=100, batch_size=8, verbose=2)

    # Evaluamos el modelo
    metrics_train = model.evaluate(X[train], y[train])
    metrics_test = model.evaluate(X[test], y[test])

    folds_train.append(metrics_train[1])
    folds_test.append(metrics_test[1])

# Imprimimos los resultados
for i in range(len(folds_train)):
    print("{} - Accuracy - Train: {:0.2f}% - Test: {:0.2f}%"
          .format((i + 1), folds_train[i] * 100, folds_test[i] * 100))

print("\nTRAIN:\n\tAVG Accuracy: {:0.2f}%\n\tSTD Accuracy: {:0.2f}%"
      .format(np.mean(folds_train) * 100, np.std(folds_train) * 100))
print("\nTEST:\n\tAVG Accuracy: {:0.2f}%\n\tSTD Accuracy: {:0.2f}%"
      .format(np.mean(folds_test) * 100, np.std(folds_test) * 100))
