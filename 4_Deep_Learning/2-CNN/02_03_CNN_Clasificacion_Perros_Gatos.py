# -*- coding: utf-8 -*-
__author__ = 'RicardoMoya'

import os
import datetime

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard

# CONSTANTES:
PIXELES = 64  # Pixele del alto y ancho de la imagen p.e-> (64,64)
BATCH_SIZE = 32  # Número de imagenes por batch
NUM_BATCHES_PER_EPOCH = 500  # Número de batches por epoch
NUM_EPOCHS = 10  # Número de epochs

# Definimos como modificar de manera aleatoria las imagenes (pixeles) de entrenamiento
#   https://keras.io/preprocessing/image/
#   rescale = normalizamos los pixeles
#   shear_range = rango de modificación aleatorio
#   zoom_range = rango de zoom aleatorio
#   horizontal_flip = Giro aleatorio de las imagenes
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

# Definimos como modificar las imagenes (pixeles) de test
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Definimos como son nuestras imagenes de entrenamiento y test
#   directory = ruta donde se encuentran las imagenes (una clase por carpeta)
#   target_size = tamaño de las imagenes 64x64. Se redimensionan a ese tamaño
#   batch_size = Nº de imagenes tras la que se calcularán los pesos de la res
#   class_mode = tipo de clasificación: binaria

train_generator = train_datagen.flow_from_directory(directory='../data/imgs_perros_gatos/training_set',
                                                    target_size=(PIXELES, PIXELES),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(directory='../data/imgs_perros_gatos/test_set',
                                                        target_size=(PIXELES, PIXELES),
                                                        batch_size=BATCH_SIZE,
                                                        class_mode='binary')

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
                 input_shape=(PIXELES, PIXELES, 3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64,
                 kernel_size=(3, 3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Imprimimos por pantalla la arquitectura de la red definida
print(model.summary())

# Compilamos el modelo
#   Función de perdida: Cross Entropy. Al ser un problema de clasificación múltiple: categorical_crossentropy
#   Optimizador: ADAM
#   Métricas a monitorizar: Accuracy
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Configuramos el Tensorboard
# Instanciar el Tensorboard: >> tensorboard --logdir ./tensorboard_logs/
# Abrir en el navegador: http://localhost:6006/
now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs('tensorboard_logs/{}'.format(now))
path = os.path.join('tensorboard_logs', now)
tsb = TensorBoard(log_dir=path, histogram_freq=1, write_graph=True, write_images=True)

# Ajuste del Modelo
#   epochs = numero de epochs
#   steps_per_epoch = Número de batches por epoch
#   validation_steps = Número de imagenes a validar por epoch
#   validation_data = Imagenes de test
model.fit_generator(train_generator,
                    epochs=NUM_EPOCHS,
                    steps_per_epoch=NUM_BATCHES_PER_EPOCH,
                    validation_steps=200,
                    validation_data=validation_generator,
                    callbacks=[tsb],
                    verbose=1)
