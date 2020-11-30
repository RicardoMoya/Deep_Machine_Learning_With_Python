# -*- coding: utf-8 -*-
__author__ = 'RicardoMoya'

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    roc_curve

# Cargamos los datos
df = pd.read_csv("../data/diabetes.csv")
print(df.sample(5))

# Cambio de Estructura de datos a Numpy
x_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
              'DiabetesPedigreeFunction', 'Age']
X = df[x_features].values
y = df['Outcome'].values

# División de datos en entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Definimos la Arquitectura de la Red Neuronal
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))  # Capa oculta 1 con 12 Neuronas y función de activación RELU
model.add(Dense(8, activation='relu'))  # Capa oculta 2 con 8 Neuronas y función de activación RELU
model.add(Dense(1, activation='sigmoid'))  # Capa de Salida de 1 Neurona con función de activación SIGMOID

# Compilamos el modelo
#   Función de perdida: Cross Entropy. Al ser un problema de clasificación binaria: binary_crossentropy
#   Optimizador: ADAM
#   Métricas a monitorizar: Accuracy, Precision, Recall y F1
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Ajuste del Modelo
#   Epochs = 200
#   Batch Size = 8
model.fit(X_train, y_train, epochs=200, batch_size=8, verbose=2)

# Obtención de predicciones y probabilidades
y_predict_train = model.predict_classes(X_train)
y_predict_test = model.predict_classes(X_test)
y_prob_train = model.predict_proba(X_train)
y_prob_test = model.predict_proba(X_test)

# Cálculo de métricas de Evaluación
metrics = list()
metrics.append({'name': 'Accuracy',
                'train': accuracy_score(y_true=y_train, y_pred=y_predict_train),
                'test': accuracy_score(y_true=y_test, y_pred=y_predict_test)})
metrics.append({'name': 'Precision',
                'train': precision_score(y_true=y_train, y_pred=y_predict_train),
                'test': precision_score(y_true=y_test, y_pred=y_predict_test)})
metrics.append({'name': 'Recall',
                'train': recall_score(y_true=y_train, y_pred=y_predict_train),
                'test': recall_score(y_true=y_test, y_pred=y_predict_test)})
metrics.append({'name': 'F1',
                'train': f1_score(y_true=y_train, y_pred=y_predict_train),
                'test': f1_score(y_true=y_test, y_pred=y_predict_test)})
metrics.append({'name': 'AUC_ROC',
                'train': roc_auc_score(y_true=y_train, y_score=y_prob_train),
                'test': roc_auc_score(y_true=y_test, y_score=y_prob_test)})

# Pasamos los resultados a un DataFrame para visualizarlos mejor
print("\nRESULTADOS")
df = pd.DataFrame.from_dict(metrics)
df.set_index("name", inplace=True)
print(df)


# Matrices de Confusión
def plot_confusion_matrix(cm, classes, title, cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Pintamos Matrices de confusión
print("Pintando Matrices de Confusión")
plt.figure(figsize=(15, 8))
plt.subplot(1, 2, 1)
plot_confusion_matrix(confusion_matrix(y_true=y_train, y_pred=y_predict_train),
                      classes=['0', '1'],
                      title='Matriz de Confusión Datos Entrenamiento')
plt.subplot(1, 2, 2)
plot_confusion_matrix(confusion_matrix(y_true=y_test, y_pred=y_predict_test),
                      classes=['0', '1'],
                      title='Matriz de Confusión Datos Test')
plt.show()

# Pintamos Curvas ROC
train_base_fpr, train_base_tpr, train_base_thresholds = roc_curve(y_train, [0 for i in range(len(y_train))])
test_base_fpr, test_base_tpr, test_base_thresholds = roc_curve(y_test, [0 for i in range(len(y_test))])
train_fpr, train_tpr, train_thresholds = roc_curve(y_train, y_prob_train)
test_fpr, test_tpr, test_thresholds = roc_curve(y_test, y_prob_test)
print("Pintando las curvas ROC")
plt.figure(figsize=(15, 8))
# TRAIN
plt.subplot(1, 2, 1)
plt.title("ROC TRAIN")
plt.plot(train_fpr, train_tpr, label='Clasificador - ROC AUC = {:0.2f}'
         .format(roc_auc_score(y_true=y_train, y_score=y_prob_train)))
plt.plot(train_base_fpr, train_base_tpr, linestyle='--', label='Clasificador Base - ROC AUC = {:0.2f}'
         .format(roc_auc_score(y_true=y_train, y_score=[0 for i in range(len(y_train))])))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')

# TEST
plt.subplot(1, 2, 2)
plt.title("ROC TEST")
plt.plot(test_fpr, test_tpr, label='Clasificador - ROC AUC = {:0.2f}'
         .format(roc_auc_score(y_true=y_test, y_score=y_prob_test)))
plt.plot(test_base_fpr, test_base_tpr, linestyle='--', label='Clasificador Base - ROC AUC = {:0.2f}'
         .format(roc_auc_score(y_true=y_test, y_score=[0 for i in range(len(y_test))])))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')

plt.show()
