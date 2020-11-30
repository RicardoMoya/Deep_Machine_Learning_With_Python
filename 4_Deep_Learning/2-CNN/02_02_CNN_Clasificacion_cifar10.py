# -*- coding: utf-8 -*-
__author__ = 'RicardoMoya'

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

# Cargamos los datos
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Cambiamos el tamaño de las imagenes a 32x32 [samples][width][height][channels]
X_train = X_train.reshape(X_train.shape[0], 32, 32, 3).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 32, 32, 3).astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# Pasamos la salida a un array (una posición por neurona de salida)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Definimos el modelo
#   Imagenes de Entrada 32 pixeles Ancho, 32 Pixeles de Alto, 3 Canales (R-G-B)
#   Capa Convolucional: 32 filtros, kernel (3x3), Función Activación RELU
#   MaxPooling: Reducción de (2,2)
#   Capa Convolucional: 64 filtros, kernel (3x3), Función Activación RELU
#   MaxPooling: Reducción de (2,2)
#   Capa Flatten: Capa de entrada del clasificador. Pasa cada Pixel a neurona
#   Capa Oculta 1: 512 Neurona, Función Activación RELU
#   Capa Oculta 2: 128 Neurona, Función Activación RELU
#   Capa Oculta 3: 32 Neurona, Función Activación RELU
#   Capa Salida: 10 Neurona (10 Clases), Función Activación SOFTMAX
model = Sequential()
model.add(Conv2D(filters=32,
                 kernel_size=(3, 3),
                 input_shape=(32, 32, 3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64,
                 kernel_size=(3, 3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

# Imprimimos por pantalla la arquitectura de la red definida
print(model.summary())

# Compilamos el modelo
#   Función de perdida: Cross Entropy. Al ser un problema de clasificación múltiple: categorical_crossentropy
#   Optimizador: ADAM
#   Métricas a monitorizar: Accuracy
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Ajuste del Modelo
#   Validation_data = Conjunto de datos de test
#   Epochs = 10
#   Batch Size = 512
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=512, verbose=1)

# EVALUACION DEL MODELO
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))
