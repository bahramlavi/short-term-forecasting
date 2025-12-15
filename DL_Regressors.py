"""
Author: Bahram Lavi, João Roberto Bertini Junior
Description: This script is the implimentation of deep learning models considered in the paper:
"Comparing Shallow and Deep Learning Regression Methods to Forecast Short-Term Oil, Water, and Gas Rates in a Pre-Salt Petroleum Field"
"""
import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.optimizers.legacy import Adam, SGD
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Flatten
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from scikeras.wrappers import KerasRegressor
from tcn import TCN

tf.reduce_retracing = True

# Config your GPU device to train DL models
device = '/GPU:0'
print(f'*** Configured GPU device {device}. \n *** You may confige this device name at DL_Regressors.py file. ') 
# Initializing the models hyperparameters treating as a set of continues values

# number of neurons for use of lstm
neurons = [5, 10, 15, 20, 50, 100]

# 1st dense layer
neurons1 = [5, 10, 15, 20, 50, 100]
# 2nd dense layer
neurons2 = [3, 5, 7]

# Optimizers for grid-search
optimizer = [SGD(), Adam()]

# filtering the maps for use of CNN
filter1 = [3, 5, 7]
filter2 = [3, 5, 7]

# some extra parameters for DL methods
weight_constraint = [1.0, 3.0]
dropout_rate = [0.25]
activation = ['relu','tanh']
batch_size = [128]
epochs = [50, 100]

    
def call_DL_Regressor():
    def get_gridsearch_with_LSTM(train_X, test_X, train_y, num_split):
        # Function to create model, required for KerasClassifier
        def create_model(neurons):
            # create model
            model = Sequential()
            model.add(LSTM(neurons, input_shape=(train_X.shape[1], train_X.shape[2])))
            model.add(Dense(1))
            return model

        # time series inside gridsearch
        tscv = TimeSeriesSplit(n_splits=num_split)
        model = KerasRegressor(model=create_model, loss='mae', verbose=0, shuffle=False)

        param_grid = dict(model__neurons=neurons,
                          batch_size=batch_size,
                          epochs=epochs,
                          optimizer=optimizer)

        grid = GridSearchCV(estimator=model,
                            param_grid=param_grid,
                            scoring='neg_mean_absolute_error',
                            n_jobs=1,
                            cv=tscv,
                            error_score='raise')  # accuracy

        train_start = datetime.datetime.now()
        with tf.device(device):
            grid_result = grid.fit(train_X, train_y)
        train_end = datetime.datetime.now()

        oilPred = grid_result.best_estimator_.predict(train_X)
        oilForecast = grid_result.best_estimator_.predict(test_X)

        reg_f = grid_result.best_estimator_.fit(train_X, train_y)
        oilPred = reg_f.predict(train_X)
        oilForecast = reg_f.predict(test_X)
        
        return oilPred,oilForecast, (train_end - train_start)

    def get_gridsearch_with_LSTMdense(train_X, test_X, train_y, num_split):
        # Function to create model, required for KerasClassifier
        def create_model(neurons1, neurons2, dp_rate):
            # create model
            model = Sequential()
            model.add(LSTM(neurons1, input_shape=(train_X.shape[1], train_X.shape[2])))
            model.add(Dropout(dp_rate))
            model.add(Dense(neurons2))
            model.add(Dense(1))
            return model

        # time series inside gridsearch
        tscv = TimeSeriesSplit(n_splits=num_split)
        model = KerasRegressor(model=create_model, loss="mae", verbose=0, shuffle=False)

        param_grid = dict(model__neurons1=neurons1,
                          model__neurons2=neurons2,
                          model__dp_rate=dropout_rate,
                          optimizer=optimizer)

        grid = GridSearchCV(estimator=model,
                            param_grid=param_grid,
                            scoring='neg_mean_absolute_error',
                            n_jobs=1,
                            cv=tscv,
                            error_score='raise')  # accuracy
        train_start = datetime.datetime.now()
        with tf.device(device):
            grid_result = grid.fit(train_X, train_y)
        train_end = datetime.datetime.now()
        oilPred = grid_result.best_estimator_.predict(train_X)
        oilForecast = grid_result.best_estimator_.predict(test_X)
        
        reg_f = grid_result.best_estimator_.fit(train_X, train_y)
        oilPred = reg_f.predict(train_X)
        oilForecast = reg_f.predict(test_X)
        
        return oilPred,oilForecast, (train_end - train_start)

    def get_gridsearch_with_LSTMbidirectional(train_X, test_X, train_y, num_split):
        # Function to create model, required for KerasClassifier
        def create_model(neurons):
            # create model
            model = Sequential()
            model.add(
                Bidirectional(LSTM(neurons), input_shape=(train_X.shape[1], train_X.shape[2])))  # return_sequences=True
            model.add(Dense(1))
            return model

        # time series inside gridsearch
        tscv = TimeSeriesSplit(n_splits=num_split)
        model = KerasRegressor(model=create_model, loss="mae", verbose=0, shuffle=False)

        #     model.run_eagerly=True
        param_grid = dict(model__neurons=neurons,
                          batch_size=batch_size,
                          epochs=epochs,
                          optimizer=optimizer)

        grid = GridSearchCV(estimator=model,
                            param_grid=param_grid,
                            scoring='neg_mean_absolute_error',
                            n_jobs=1,
                            cv=tscv,
                            error_score='raise')  # accuracy
        train_start = datetime.datetime.now()
        with tf.device(device):
            grid_result = grid.fit(train_X, train_y)
        train_end = datetime.datetime.now()

        oilPred = grid_result.best_estimator_.predict(train_X)
        oilForecast = grid_result.best_estimator_.predict(test_X)
        
        reg_f = grid_result.best_estimator_.fit(train_X, train_y)
        oilPred = reg_f.predict(train_X)
        oilForecast = reg_f.predict(test_X)
        
        return oilPred,oilForecast, (train_end - train_start)

    def get_gridsearch_with_LSTMbidiDense(train_X, test_X, train_y, num_split):
        def create_model(neurons1, neurons2, dp_rate):
            # create model
            model = Sequential()
            model.add(Bidirectional(LSTM(neurons1), input_shape=(train_X.shape[1], train_X.shape[2])))
            model.add(Dropout(dp_rate))
            model.add(Dense(neurons2))
            model.add(Dense(1))
            return model

        # time series inside gridsearch
        tscv = TimeSeriesSplit(n_splits=num_split)
        model = KerasRegressor(model=create_model, loss="mae", verbose=0, shuffle=False)

        param_grid = dict(model__neurons1=neurons1,
                          model__neurons2=neurons2,
                          model__dp_rate=dropout_rate,
                          batch_size=batch_size,
                          epochs=epochs,
                          optimizer=optimizer)

        grid = GridSearchCV(estimator=model,
                            param_grid=param_grid,
                            scoring='neg_mean_absolute_error',
                            n_jobs=1,
                            cv=tscv,
                            error_score='raise')  # accuracy
        train_start = datetime.datetime.now()
        with tf.device(device):
            grid_result = grid.fit(train_X, train_y)
        train_end = datetime.datetime.now()
        oilPred = grid_result.best_estimator_.predict(train_X)
        oilForecast = grid_result.best_estimator_.predict(test_X)
        
        reg_f = grid_result.best_estimator_.fit(train_X, train_y)
        oilPred = reg_f.predict(train_X)
        oilForecast = reg_f.predict(test_X)
        
        return oilPred,oilForecast, (train_end - train_start)

    def get_gridsearch_with_CNN1D(train_X, test_X, train_y, num_split):
        n_timesteps, n_features = train_X.shape[1], train_X.shape[2]

        def create_model(filter1, filter2, neurons2, act, dp_rate):
            # create model
            model = Sequential()
            model.add(Conv1D(filters=filter1, kernel_size=4, activation=act, input_shape=(n_timesteps, n_features),
                             padding='same'))  # ks = 3
            model.add(Conv1D(filters=filter2, kernel_size=2, activation=act, padding='same'))
            model.add(Dropout(dp_rate))
            model.add(MaxPooling1D(pool_size=2, padding='same'))  # pool_size=2
            model.add(Flatten())
            model.add(Dense(neurons2, activation=act))
            model.add(Dense(1))
            return model

        tscv = TimeSeriesSplit(n_splits=num_split)
        model = KerasRegressor(model=create_model, loss="mae", verbose=0, shuffle=False)

        param_grid = dict(model__neurons2=neurons2,
                          model__filter1=filter1,
                          model__filter2=filter2,
                          model__dp_rate=dropout_rate,
                          model__act=activation,
                          batch_size=batch_size,
                          epochs=epochs,
                          optimizer=optimizer)

        grid = GridSearchCV(estimator=model,
                            param_grid=param_grid,
                            scoring='neg_mean_absolute_error',
                            n_jobs=1,
                            cv=tscv,
                            error_score='raise')  # accuracy
        train_start = datetime.datetime.now()
        with tf.device(device):
            grid_result = grid.fit(train_X, train_y)
        train_end = datetime.datetime.now()
        oilPred = grid_result.best_estimator_.predict(train_X)
        oilForecast = grid_result.best_estimator_.predict(test_X)
        
        reg_f = grid_result.best_estimator_.fit(train_X, train_y)
        oilPred = reg_f.predict(train_X)
        oilForecast = reg_f.predict(test_X)
        
        return oilPred,oilForecast, (train_end - train_start)

    def get_gridsearch_with_CNNLSTM(train_X, test_X, train_y, num_split):
        n_timesteps, n_features = train_X.shape[1], train_X.shape[2]

        def create_model(filter1, neurons1, act, dp_rate):
            # create model
            model = Sequential()
            model.add(Conv1D(filters=filter1, kernel_size=3, activation=act, input_shape=(n_timesteps, n_features),
                             padding='same'))  # ks = 3  padding='same'
            model.add(Dropout(dp_rate))
            # model.add(Flatten())
            # model.add(MaxPooling1D(pool_size=2))  # pool_size=2
            model.add(LSTM(neurons1))
            model.add(Dense(1))
            return model

        tscv = TimeSeriesSplit(n_splits=num_split)
        model = KerasRegressor(model=create_model, loss="mae", verbose=0, shuffle=False)

        param_grid = dict(model__neurons1=neurons1,
                          model__filter1=filter1,
                          model__dp_rate=dropout_rate,
                          model__act=activation,
                          batch_size=batch_size,
                          epochs=epochs,
                          optimizer=optimizer)

        grid = GridSearchCV(estimator=model,
                            param_grid=param_grid,
                            scoring='neg_mean_absolute_error',
                            n_jobs=1,
                            cv=tscv,
                            error_score='raise')  # accuracy
        train_start = datetime.datetime.now()
        with tf.device(device):
            grid_result = grid.fit(train_X, train_y)
        train_end = datetime.datetime.now()

        oilPred = grid_result.best_estimator_.predict(train_X)
        oilForecast = grid_result.best_estimator_.predict(test_X)
        
        reg_f = grid_result.best_estimator_.fit(train_X, train_y)
        oilPred = reg_f.predict(train_X)
        oilForecast = reg_f.predict(test_X)
        
        return oilPred,oilForecast, (train_end - train_start)

    def get_gridsearch_with_TCN(train_X, test_X, train_y, num_split):

        def create_model(neurons, kernel_size):
            # create model
            model = Sequential()
            model.add(
                TCN(
                    nb_filters=neurons,        
                    kernel_size=kernel_size,   
                    dilations=(2, 4, 8),   
                    return_sequences=False,    
                    input_shape=(train_X.shape[1], train_X.shape[2])
                )
            )
            
            model.add(Dense(1))
            
            model.compile(optimizer='adam', loss='mae') 
            return model

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=num_split)
        
        # Use KerasRegressor from scikeras
        model = KerasRegressor(
            model=create_model, 
            loss='mae', 
            verbose=0, 
            shuffle=False, 
            # Pass default values for TCN-specific params to the wrapper
            model__neurons=neurons, 
            model__kernel_size=filter1
        )

        # Define the hyperparameter grid
        param_grid = dict(
            model__neurons=neurons1,          # TCN filters
            model__kernel_size=filter1,  # TCN kernel size
            batch_size=batch_size,
            epochs=epochs,
            optimizer=optimizer
        )

        # Initialize GridSearchCV
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='neg_mean_absolute_error',
            n_jobs=1,
            cv=tscv,
            error_score='raise'
        )

        train_start = datetime.datetime.now()
        with tf.device(device):
            grid_result = grid.fit(train_X, train_y)
        train_end = datetime.datetime.now()

        oilPred = grid_result.best_estimator_.predict(train_X)
        oilForecast = grid_result.best_estimator_.predict(test_X)

        reg_f = grid_result.best_estimator_.fit(train_X, train_y)
        oilPred = reg_f.predict(train_X)
        oilForecast = reg_f.predict(test_X)
        
        return oilPred,oilForecast, (train_end - train_start)
        
    regs = {'LSTM': get_gridsearch_with_LSTM,
            'LSTMDense': get_gridsearch_with_LSTMdense,
            'BiLSTM': get_gridsearch_with_LSTMbidirectional,
            'BiLSTMDense': get_gridsearch_with_LSTMbidiDense,
            'CNN': get_gridsearch_with_CNN1D,
            'CNNLSTM': get_gridsearch_with_CNNLSTM,
            'TCN':get_gridsearch_with_TCN
            }
    return regs



