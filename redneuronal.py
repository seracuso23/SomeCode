#===========================================================================================================================================================
#Aquí te proporciono un código de ejemplo en Python utilizando TensorFlow y Keras para ajustar una red neuronal para clasificación binaria para 
#datos tabulares con clases desbalanceadas donde se optimicen la cantidad de capas y sus tamaños, así como los núcleos de cada capa:




import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Cargar los datos
df = pd.read_csv('datos.csv')

# Dividir los datos en conjuntos de entrenamiento y prueba
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calcular la proporción de clases para el ajuste del modelo
class_weight = {0: np.sum(y == 1) / len(y), 1: np.sum(y == 0) / len(y)}

# Definir el modelo
def create_model(num_layers, layer_sizes, kernel_sizes):
    model = Sequential()
    for i in range(num_layers):
        if i == 0:
            model.add(Dense(layer_sizes[i], activation='relu', input_shape=(X_train.shape[1],)))
        else:
            model.add(Dense(layer_sizes[i], activation='relu'))
        model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Probar diferentes combinaciones de capas y tamaños
num_layers = [2, 3, 4]
layer_sizes = [[128, 64], [256, 128, 64], [512, 256, 128, 64]]

for i in range(len(num_layers)):
    for j in range(len(layer_sizes)):
        model = create_model(num_layers[i], layer_sizes[j], kernel_sizes)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, class_weight=class_weight, callbacks=[es])
        y_pred = model.predict(X_test)
        y_pred_classes = np.round(y_pred)
        print(f'Layers: {num_layers[i]}, Layer Sizes: {layer_sizes[j]}')
        print(classification_report(y_test, y_pred_classes))

#===========================================================================================================================================================
#Aquí tienes un código de ejemplo en Python utilizando TensorFlow, Keras y Optuna para ajustar una red neuronal para clasificación binaria para datos 
#tabulares con clases desbalanceadas donde se optimicen la cantidad de capas y sus tamaños, así como los núcleos de cada capa:

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import optuna
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Cargar los datos
df = pd.read_csv('datos.csv')

# Dividir los datos en conjuntos de entrenamiento y prueba
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calcular la proporción de clases para el ajuste del modelo
class_weight = {0: np.sum(y == 1) / len(y), 1: np.sum(y == 0) / len(y)}

# Definir la función de objetivo para Optuna
def objective(trial):
    num_layers = trial.suggest_int('num_layers', 2, 4)
    layer_sizes = []
    for i in range(num_layers):
        layer_sizes.append(trial.suggest_int(f'layer_{i}_size', 32, 512))
    kernel_sizes = trial.suggest_int('kernel_size', 2, 5)

    # Definir el modelo
    model = Sequential()
    for i in range(num_layers):
        if i == 0:
            model.add(Dense(layer_sizes[i], activation='relu', input_shape=(X_train.shape[1],)))
        else:
            model.add(Dense(layer_sizes[i], activation='relu'))
        model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, class_weight=class_weight, callbacks=[es])
    y_pred = model.predict(X_test)
    y_pred_classes = np.round(y_pred)
    score = classification_report(y_test, y_pred_classes, output_dict=True)['weighted avg']['f1-score']
    return score

# Ejecutar Optuna para encontrar la combinación óptima de capas y tamaños
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f'Best score: {study.best_value}')
print(f'Best params: {study.best_params}')


#===========================================================================================================================================================

#Aquí tienes un código de ejemplo en Python que utiliza TensorFlow, Keras, Optuna y validación cruzada para ajustar una red neuronal para 
#clasificación binaria para datos tabulares con clases desbalanceadas, optimizando la cantidad de capas, sus tamaños, las funciones de activación, learning rate 
#y dropout utilizando Optuna:

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
import optuna
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Cargar los datos
df = pd.read_csv('datos.csv')

# Dividir los datos en conjuntos de entrenamiento y prueba
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calcular la proporción de clases para el ajuste del modelo
class_weight = {0: np.sum(y == 1) / len(y), 1: np.sum(y == 0) / len(y)}

# Definir la función de objetivo para Optuna
def objective(trial):
    num_layers = trial.suggest_int('num_layers', 2, 4)
    layer_sizes = []
    activations = []
    for i in range(num_layers):
        layer_sizes.append(trial.suggest_int(f'layer_{i}_size', 32, 512))
        activations.append(trial.suggest_categorical(f'layer_{i}_activation', ['relu', 'sigmoid']))
    dropout_rates = []
    for i in range(num_layers - 1):
        dropout_rates.append(trial.suggest_float(f'layer_{i}_dropout', 0.0, 0.5))
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)

    # Definir el modelo
    model = Sequential()
    for i in range(num_layers):
        if i == 0:
            model.add(Dense(layer_sizes[i], activation=activations[i], input_shape=(X_train.shape[1],)))
        else:
            model.add(Dense(layer_sizes[i], activation=activations[i]))
            model.add(Dropout(dropout_rates[i-1]))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Validación cruzada
    cv = StratifiedKFold(n_splits=5)
    scores = []
    for train_index, val_index in cv.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        history = model.fit(X_train_fold, y_train_fold, epochs=100, batch_size=32, validation_data=(X_val_fold, y_val_fold),
                            class_weight=class_weight, callbacks=[es])
        y_pred = model.predict(X_test)
        y_pred_classes = np.round(y_pred)
        score = classification_report(y_test, y_pred_classes, output_dict=True)['weighted avg']['f1-score']
        scores.append(score)
    return np.mean(scores)

# Ejecutar Optuna para encontrar la combinación óptima de capas y tamaños
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f'Best score: {study.best_value}')
print(f'Best params: {study.best_params}')

#===========================================================================================================================================================

#A continuación te proporciono un código que puedes utilizar para ajustar una red neuronal para clasificación binaria con datos tabulares con clases 
#desbalanceadas utilizando optuna para optimizar la cantidad de capas, sus tamaños y para cada capa las funciones de activación, optimizador, learning rate 
#y dropout mediante validación cruzada.

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
import optuna

# Cargar los datos
df = pd.read_csv('datos.csv')

# Dividir los datos en características y etiquetas
X = df.drop('etiqueta', axis=1).values
y = df['etiqueta'].values

# Definir la función de objetivo
def objective(trial):

    # Definir los hiperparámetros a optimizar
    n_layers = trial.suggest_int('n_layers', 1, 5)
    layers = []
    for i in range(n_layers):
        layers.append(trial.suggest_int(f'layer_{i}_size', 32, 512))

    activation = trial.suggest_categorical('activation', ['relu', 'sigmoid', 'tanh'])
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 1.0)

    # Definir la arquitectura del modelo
    model = keras.Sequential()
    for i in range(n_layers):
        if i == 0:
            model.add(keras.layers.Dense(layers[i], activation=activation, input_dim=X.shape[1]))
            model.add(keras.layers.Dropout(dropout_rate))
        else:
            model.add(keras.layers.Dense(layers[i], activation=activation))
            model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    # Compilar el modelo
    if optimizer == 'adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        opt = keras.optimizers.SGD(learning_rate=learning_rate)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    # Realizar validación cruzada
    epochs_p = trial.suggest_int("epochs", 10, 50)
    batch_size_p = trial.suggest_int("batch_size", 16, 256)
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    for train_idx, test_idx in kfold.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        model.fit(X_train, y_train, epochs=epochs_p, batch_size=batch_size_p, verbose=0)
        _, accuracy = model.evaluate(X_test, y_test, verbose=0)
        accuracies.append(accuracy)

    # Devolver la media de las exactitudes obtenidas en la validación cruzada
    return np.mean(accuracies)

# Ejecutar la optimización con Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Imprimir los resultados
print('Mejor valor de exactitud obtenido: ', study.best_value)
print('Mejores hiperparámetros encontrados: ', study.best_params)




import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MultiLabelBinarizer


def generate_class_weights(class_series, multi_class=True, one_hot_encoded=False):
  """
  Method to generate class weights given a set of multi-class or multi-label labels, both one-hot-encoded or not.
  Some examples of different formats of class_series and their outputs are:
    - generate_class_weights(['mango', 'lemon', 'banana', 'mango'], multi_class=True, one_hot_encoded=False)
    {'banana': 1.3333333333333333, 'lemon': 1.3333333333333333, 'mango': 0.6666666666666666}
    - generate_class_weights([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], multi_class=True, one_hot_encoded=True)
    {0: 0.6666666666666666, 1: 1.3333333333333333, 2: 1.3333333333333333}
    - generate_class_weights([['mango', 'lemon'], ['mango'], ['lemon', 'banana'], ['lemon']], multi_class=False, one_hot_encoded=False)
    {'banana': 1.3333333333333333, 'lemon': 0.4444444444444444, 'mango': 0.6666666666666666}
    - generate_class_weights([[0, 1, 1], [0, 0, 1], [1, 1, 0], [0, 1, 0]], multi_class=False, one_hot_encoded=True)
    {0: 1.3333333333333333, 1: 0.4444444444444444, 2: 0.6666666666666666}
  The output is a dictionary in the format { class_label: class_weight }. In case the input is one hot encoded, the class_label would be index
  of appareance of the label when the dataset was processed. 
  In multi_class this is np.unique(class_series) and in multi-label np.unique(np.concatenate(class_series)).
  Author: Angel Igareta (angel@igareta.com)
  """
  if multi_class:
    # If class is one hot encoded, transform to categorical labels to use compute_class_weight   
    if one_hot_encoded:
      class_series = np.argmax(class_series, axis=1)
  
    # Compute class weights with sklearn method
    class_labels = np.unique(class_series)
    class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=class_series)
    return dict(zip(class_labels, class_weights))
  else:
    # It is neccessary that the multi-label values are one-hot encoded
    mlb = None
    if not one_hot_encoded:
      mlb = MultiLabelBinarizer()
      class_series = mlb.fit_transform(class_series)

    n_samples = len(class_series)
    n_classes = len(class_series[0])

    # Count each class frequency
    class_count = [0] * n_classes
    for classes in class_series:
        for index in range(n_classes):
            if classes[index] != 0:
                class_count[index] += 1
    
    # Compute class weights using balanced method
    class_weights = [n_samples / (n_classes * freq) if freq > 0 else 1 for freq in class_count]
    class_labels = range(len(class_weights)) if mlb is None else mlb.classes_
    return dict(zip(class_labels, class_weights))
