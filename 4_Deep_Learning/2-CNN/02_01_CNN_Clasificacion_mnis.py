# -*- coding: utf-8 -*-
__author__ = 'RicardoMoya'

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

# Cargamos los datos
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Cambiamos el tamaño de las imagenes a 28x28 [samples][width][height][channels]
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# normalizamos la matriz de imagenes de 0-255 a 0-1
X_train = X_train / 255
X_test = X_test / 255

# Pasamos la salida a un array (una posición por neurona de salida)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# Definimos el modelo
#   Imagenes de Entrada 28 pixeles Ancho, 28 Pixeles de Alto, 1 Canal
#   Capa Convolucional: 32 filtros, kernel (3x3), Función Activación RELU
#   MaxPooling: Reducción de (2,2)
#   Capa Convolucional: 64 filtros, kernel (3x3), Función Activación RELU
#   MaxPooling: Reducción de (2,2)
#   Capa Flatten: Capa de entrada del clasificador. Pasa cada Pixel a neurona
#   Capa Oculta 1: 128 Neurona, Función Activación RELU
#   Capa Oculta 2: 50 Neurona, Función Activación RELU
#   Capa Salida: 10 Neurona (10 Clases), Función Activación SOFTMAX
model = Sequential()
model.add(Conv2D(filters=32,
                 kernel_size=(3, 3),
                 input_shape=(28, 28, 1),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64,
                 kernel_size=(3, 3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(50, activation='relu'))
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
#   Epochs = 5
#   Batch Size = 512
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=512, verbose=1)

# Evaluamos el modelo
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100 - scores[1] * 100))
