import optuna
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


# Paso 1: cargar y preparar los datos
X, y = load_iris(return_X_y=True)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Paso 2: definir la función objetivo para Optuna
def objective(trial):
    # Definir el espacio de búsqueda para cada hiperparámetro
    n_layers = trial.suggest_int('n_layers', 1, 4)
    hidden_layer_sizes = []
    for i in range(n_layers):
        layer_size = trial.suggest_int(f"n_units_l{i+1}", 1, 100)
        hidden_layer_sizes.append(layer_size)
    activation_funcs = []
    for i in range(n_layers):
        activation_func = trial.suggest_categorical(f"activation_l{i+1}", ['relu', 'tanh', 'sigmoid', 'softsign', 'softmax'])
        activation_funcs.append(activation_func)
    dropouts = []
    for i in range(n_layers):
        dropout = trial.suggest_uniform(f"dropout_l{i+1}", 0, 1)
        dropouts.append(dropout)
    epoch = trial.suggest_int('epoch', 10, 100)
    batch_size = trial.suggest_int('batch_size', 16, 128)
    optimizer = trial.suggest_categorical('optimizer', ['SGD', 'Adam', 'Adagrad', 'AdaDelta', 'RMSprop'])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)

    # Definir el modelo MLP con los hiperparámetros sugeridos por Optuna
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation_funcs,
                        alpha=0.0001, batch_size=batch_size, learning_rate=learning_rate,
                        learning_rate_init=0.001, max_iter=epoch, solver=optimizer, tol=0.0001,
                        validation_fraction=0.1, verbose=False, warm_start=False, early_stopping=True,
                        n_iter_no_change=5, random_state=42)

    # Calcular la precisión promedio mediante validación cruzada de 5 veces
    score = cross_val_score(mlp, X_train, y_train, cv=5, n_jobs=-1).mean()
    return score


# Paso 3: optimizar los hiperparámetros con Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Paso 4: imprimir los mejores hiperparámetros y el puntaje promedio en el conjunto de prueba
best_params = study.best_params
print(f"Los mejores hiperparámetros son: {best_params}")

mlp = MLPClassifier(hidden_layer_sizes=[best_params[f"n_units_l{i+1}"] for i in range(best_params["n_layers"])],
                    activation=[best_params[f"activation_l{i+1}"] for i in range(best_params["n_layers"])],
                    alpha=0.0001, batch_size=best
