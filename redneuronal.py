import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
import optuna
from sklearn.model_selection import KFold
import numpy as np

def create_model(trial):
    model = keras.Sequential()
    n_layers = trial.suggest_int('n_layers', 1, 3)
    activation = trial.suggest_categorical('activation', ['relu', 'sigmoid', 'tanh'])
    model.add(Dense(trial.suggest_int('n_units', 32, 512), activation=activation, input_shape=(input_shape,)))
    model.add(Dropout(trial.suggest_float('dropout', 0.0, 0.5)))
    for i in range(n_layers):
        model.add(Dense(trial.suggest_int(f'n_units_l{i}', 32, 512), activation=activation))
        model.add(Dropout(trial.suggest_float(f'dropout_l{i}', 0.0, 0.5)))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def objective(trial):
    kfold = KFold(n_splits=5, shuffle=True)
    val_losses = []
    for train_idx, val_idx in kfold.split(X_train, y_train):
        X_train_fold = X_train[train_idx]
        y_train_fold = y_train[train_idx]
        X_val_fold = X_train[val_idx]
        y_val_fold = y_train[val_idx]
        model = create_model(trial)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train_fold, y_train_fold, epochs=trial.suggest_int('epochs', 10, 50), batch_size=trial.suggest_int('batch_size', 32, 256), verbose=0)
        val_loss, val_acc = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        val_losses.append(val_loss)
    return np.mean(val_losses)

# Define the search space and optimization algorithm using Optuna
input_shape = X_train.shape[1]
num_classes = y_train.shape[1]
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Train the model with the best hyperparameters found by Optuna
best_trial = study.best_trial
best_model = create_model(best_trial)
best_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
best_model.fit(X_train, y_train, epochs=best_trial.params['epochs'], batch_size=best_trial.params['batch_size'], verbose=0)

# Evaluate the model on the test set
test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
print(f'Test loss: {test_loss}, Test accuracy: {test_acc}')
