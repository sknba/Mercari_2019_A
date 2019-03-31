
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers
import pandas as pd
import numpy as np

#Building the ANN
regressor = Sequential()
regressor.add(Dense(units = 500, kernel_initializer = 'uniform', activation = 'relu', input_dim = 1032))
regressor.add(Dropout(rate = 0.1))
regressor.add(Dense(units = 250, kernel_initializer = 'uniform', activation = 'relu'))
regressor.add(Dropout(rate = 0.1))
regressor.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'relu'))
regressor.compile(optimizer = 'sgd', loss = 'mean_squared_error', metrics = ['mean_absolute_error'])

#training the ANN
chunk_number = 0

for training_set in pd.read_csv('train_preprocessed.csv', chunksize = 10000):                                 
    print('Chunk number: ' + str(chunk_number) + ' Number of chunks ~150')
    X_train = training_set.iloc[:, :-1].values
    y_train = training_set.iloc[:, -1].values
    regressor.fit(X_train, y_train, batch_size = 64, epochs = 50)
    chunk_number = chunk_number + 1

"""Sieć działa, kod na na dole ma służyć do dobrania najlepszych parametrów dla niej
To jest dosc proste, ale wymaga sporo czasu i wolnej mocy obliczeniowej :D

def build_regressor():
    regressor = Sequential()
    regressor.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    regressor.dropout(rate = 0.2)
    regressor.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    regressor.dropout(rate = 0.2)
    regressor.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['neg_mean_squared_error'])
    return regressor
regressor = KerasRegressor(build_fn = build_regressor, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

# Tuning the ANN
def build_regressor(optimizer):
    regressor = Sequential()
    regressor.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    regressor.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    regressor.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    regressor.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['neg_mean_squared_error'])
    return regressor
regressor = KerasRegressor(build_fn = build_regressor)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = regressor,
                           param_grid = parameters,
                           scoring = 'neg_mean_squared_error',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
"""