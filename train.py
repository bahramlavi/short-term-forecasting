"""
Author: Bahram Lavi, João Roberto Bertini Junior
Description: This script is the implimentation of our paper entitled:
"Comparing Shallow and Deep Learning Regression Methods to Forecast Short-Term Oil, Water, and Gas Rates in a Pre-Salt Petroleum Field"
"""

from warnings import filterwarnings
filterwarnings('ignore')
import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import argparse
from pathlib import Path

#from sklearn.metrics import mean_absolute_error, mean_squared_error

# path to the raw well data
path_hist_data = './U4_Data/U4_HistoryData_02Sep18-02Jan27/'
path_forecast_data = './U4_Data/U4_ForecastData_03Jan27-02Jul27/'

# path to the experimental outpus
path2main_output_folder = './Experiments/'
Path(path2main_output_folder).mkdir(parents=True, exist_ok=True)
path2signal = 'Signals/'
Path(path2main_output_folder+path2signal).mkdir(parents=True, exist_ok=True)
path2csvs = 'CSVs/'
Path(path2main_output_folder+path2csvs).mkdir(parents=True, exist_ok=True)
path2figures = 'Figures'
Path(path2main_output_folder+path2figures).mkdir(parents=True, exist_ok=True)
path2bestmodels = 'Best_models'
Path(path2main_output_folder+path2bestmodels).mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()

# Add the 'model_training' argument.
# help provides a description for the user when they run the script with -h or --help.
parser.add_argument(
    '--model_training',
    type=str,
    required=True,
    help="Specify training models on ML or DL (ml for shallow ML and dl for deep learing models)"
)

# Add the 'attribute_names' argument.
# help provides a description for the user when they run the script with -h or --help.
parser.add_argument(
    '--model_input',
    nargs='+',
    required=True,
    help="A space-separated list of input feature names (e.g., for only oil rate use OR; and for oil and water rates use OR WR; all attributes use OR WR GR BHP')."
)

# Add the 'model_prediction' argument.
# required=True means this argument must be provided by the user.
parser.add_argument(
    '--model_output',
    type=str,
    required=True,
    help="The model prediction output (e.g., OR, WR, GR)."
)
args = parser.parse_args()

# define the models input feature:
# in the case of only oil rate as a single predictor use ['OR']
# in the case of only water rate as a single predictor use ['WR']
# in the case of only gas rate as a single predictor use ['GR']
attribute_names = args.model_input#['OR'] 
# in the case of using main predictor along with other attribute provided in the datset, 
#for instance using water, gas, and bottom-hole pressure consider ['OR', 'WR','GR','BHP']
# attribute_names = ['OR', 'WR', 'GR', 'BHP'] 

# Specify the ML model expected prediction output: 
# in the case of oil rates use 'OR'
# in the case of water rates use 'WR'
# in the case of gas rates use 'GR'
model_prediction = args.model_output#'OR'

train_type=args.model_training

atts_name = ''
for i in range(len(attribute_names)):
    atts_name += attribute_names[i] + '_' if i + 1 < len(attribute_names) else attribute_names[i]

# time-lagged range
lag_sizes = [2, 4, 6, 8, 10, 20]
# prediction in horizon sizes
periods_h = [7, 30, 90, 180]

# Time-series cross validation (TSCV) number of splits
number_of_splits = [2,3,5]#   [2, 3, 5]                   ##  Joao

# experimental evaluation on the well producers [11...16]
well_id = range(11, 17)
well_state = ['P']
         
history_ends=2891
initial_day_zero_dopping=1200


print(f'\033[91m***** Running Experiments on training {train_type} models with prediction windows {periods_h}.')
print(f'**Lag size is selectec on {lag_sizes}') 
print(f'**Time-Series Cross-Validation size is selectec on {number_of_splits}\033[0m') 

#Columns for final dataframe with results
df_result_column_names = ['Regressor',
                          'Well',
                          'lag',
                          'period',
                          'tscv',
                          'RMSE',
                          'MAE',
                          'WMAPE',
                          'NQDS',
                          'Proc. Time']

def compute_NQDS(y, y_hat):
    tolorance = 2
    cons = 20
    AQD = np.sum((tolorance*y+cons)**2)
    SD = np.sum((y_hat-y))
    if SD != 0:
        QDS = (SD/np.abs(SD)) * np.sum((y_hat-y))**2
    else:
        QDS = np.sum((y_hat-y))**2
    NQDS = QDS/AQD
    
    return np.round(NQDS,2)
    
# This fucntion calls the production dataset from ./U4_Data/ directory    
def load_data(well_id, wtype):

    assert np.isin(wtype, ['P'])

    def preprocess_U4_data(df, wtype='P'):
        if wtype == 'P':
            # For the data of PRODECUER
            df.rename(columns={list(df.keys())[0]: 'Dia',
                               'Unnamed: 1': 'OR',  # Oil Rate Service Charge
                               'Unnamed: 2': 'WR',  # Water Rate Service Charge
                               'Unnamed: 3': 'GR',  # Gas Rate Service Charge
                               'Unnamed: 4': 'BHP', }, inplace=True)  # Well Bottom-hole Pressure
            df.drop(0, axis=0, inplace=True)

            df['OR'] = pd.to_numeric(df['OR'])
            df['WR'] = pd.to_numeric(df['WR'])  # Water Rate SC
            df['GR'] = pd.to_numeric(df['GR'])
            df['BHP'] = pd.to_numeric(df['BHP'])  # Well Bottom-hole Pressure

        df.reset_index(inplace=True)
        del df['index']

        return df.reindex().drop('Dia', axis=1)

    file_name_hist, file_name_forecast = '', ''
    if wtype == 'P':
        file_name_hist = f'{wtype}{well_id}_HistoryData_02Sep18-02Jan27.csv'
        file_name_forecast = f'{wtype}{well_id}_U4-R_ForecastData_03Jan27-02Jul27.csv'
    hist_df = pd.read_csv(path_hist_data + file_name_hist,
                          #                               usecols=use_attributes,
                          encoding='latin1', na_values='#DIV/0!').fillna(0)
    hist_df = preprocess_U4_data(hist_df, wtype=wtype)
    forcast_df = pd.read_csv(path_forecast_data + file_name_forecast,
                             #                                 usecols=use_attributes,
                             encoding='latin1', na_values='#DIV/0!').fillna(0)
    forcast_df = preprocess_U4_data(forcast_df, wtype=wtype)
    return hist_df, forcast_df


# This function works in data pre-processing steps like lags and zero data imputations
def data_handling(data, atts, predictor, with_lags=True, impute_zeros=True, t_lags=0, dah=1):
    # Adding lag data 
    def add_lags(df):
        for lg in range(1, t_lags):
            for att in atts:
                col = f'{att}_lag_{lg}'  # .format(att, lg)  #Creates a column name
                df.insert(len(df.columns) - 1, col, pd.Series(df[att].shift(lg).values))
        return df
    # Imputing zero values only on targer variable
    def remove_zero_values(df):
        df.insert(len(df.columns), 'pred', pd.Series(df[predictor].shift(-dah)))

        def consecutive(data_n, stepsize=1):
            return np.split(data_n, np.where(np.diff(data_n) != stepsize)[0] + 1)

        z_inx = df[df['pred'] == 0].index.sort_values().values # to impute target values
        zero_list = consecutive(z_inx)

        for zrs in zero_list:
            new_value_ = (df.iloc[zrs[0] - 1]['pred'] + df.iloc[zrs[-1] + 1]['pred']) / 2.0
            for z_i in zrs:
                df.at[z_i, 'pred'] = new_value_
               
        if impute_zeros:
            zd_inx = df[df[predictor] == 0].index.sort_values().values # to impute input features
            zerod_list = consecutive(zd_inx)

            for zrsd in zerod_list:
                new_value_ = (df.iloc[zrsd[0] - 1][predictor] + df.iloc[zrsd[-1] + 1][predictor]) / 2.0
                for z_i in zrsd:
                    df.at[z_i, predictor] = new_value_

        return df
    
    data=data.loc[initial_day_zero_dopping:]

    data.reset_index(inplace=True)
    data_final = remove_zero_values(data.copy())
    if with_lags:
        data_final = add_lags(data_final)
    data_final.dropna(inplace=True, axis=0)
    
    return data_final.drop('index', axis=1)

# This function trains shallow ML models on the UNISIM-IV dataset
def train_ML_models():
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
    from scikeras.wrappers import KerasRegressor
    from ML_Regressors import call_ML_regressors
    df_res = pd.DataFrame(columns=df_result_column_names)

    regs = call_ML_regressors()
    for w_idx, w in enumerate(well_id):
        print(f'\033[91m***** Processing on Well_ID = {w} with attributes {atts_name} to predict {model_prediction} \033[0m')
        hist_df_main, forecast_df_main = load_data(well_id=w, wtype='P')
        for lg in lag_sizes:
            for prd in periods_h:
                df = pd.concat([hist_df_main, forecast_df_main])
                df = df.loc[:history_ends+prd]
                df_main = data_handling(df[attribute_names], attribute_names, predictor=model_prediction, with_lags=True, t_lags=lg, dah=prd)
                print(df_main.columns)
                values = df_main.values
                values = values.astype('float32')
                features = values[:, :-1]
                labels = values[:, -1]

                # separates train and test
                train_X = features[:-prd, :]
                test_X = features[-prd:, :]  # forecast
                train_y = labels[:-prd]
                test_y = labels[-prd:]
                print(train_X.shape,test_X.shape)

                # transform data using training
                scaler = MinMaxScaler(feature_range=(0, 1))  # (0,1)
                train_X = scaler.fit_transform(train_X)  # train_X
                test_X = scaler.transform(test_X)

                scalerLA = MinMaxScaler(feature_range=(0, 1))
                train_y = scalerLA.fit_transform(train_y.reshape(-1, 1))  #  modification 1

                for time_cv in number_of_splits:
                    tscv = TimeSeriesSplit(n_splits=time_cv)
                    for reg in regs:
                        print(f'\033[93m Regressor = {reg}, Lag = {lg}, Period = {prd}, TSCV = {time_cv} \033[0m')
                        if os.path.isfile(path2main_output_folder + path2signal + f'{w}_{atts_name}_{reg}_{lg}_{prd}_{time_cv}.npy'):
                            print('\033[91m***Existed entry***\033[0m')
                            df_res = pd.read_csv(path2main_output_folder + path2csvs + f'{atts_name}_ml_results.csv')
                            print(len(df_res))
                            continue
                        reg_info = regs[reg]
                        reg_f = reg_info['function'](**reg_info['params'])
                        train_start = datetime.datetime.now()
                        grid = GridSearchCV(estimator=reg_f,
                                            param_grid=reg_info['search_params'],
                                            scoring='neg_mean_absolute_error', n_jobs=-1, cv=tscv)  #
                        grid_result = grid.fit(train_X, train_y.ravel())
                        train_end = datetime.datetime.now()
                        
                        reg_f = reg_info['function'](**grid_result.best_estimator_.get_params())
                        reg_f.fit(train_X, train_y.ravel())
                        oilPred = reg_f.predict(train_X)
                        oilForecast = reg_f.predict(test_X)
                        
                        oilPred = scalerLA.inverse_transform(oilPred.reshape(-1, 1))
                        oilForecast = scalerLA.inverse_transform(oilForecast.reshape(-1,1))


                        rMSE = np.sqrt(mean_squared_error(test_y[:prd], oilForecast[:prd]))
                        mae = mean_absolute_error(test_y[:prd], oilForecast[:prd])

                        # Calculate the numerator and denominator
                        numerator = np.sum(np.abs(test_y[:prd] - oilForecast[:prd]))
                        denominator = np.sum(np.abs(test_y[:prd]))
                        # Calculate the scalar output
                        wmape = numerator / denominator

                        nqds=compute_NQDS(test_y[:prd], oilForecast[:prd])
                        
                        predData = np.vstack((oilPred, oilForecast[:prd]))  # 10000 ? João
                        df_res = pd.concat([df_res, pd.DataFrame([reg,
                                                                  w,
                                                                  lg,
                                                                  prd,
                                                                  time_cv,
                                                                  np.round(rMSE, 2),
                                                                  np.round(mae, 2),
                                                                  np.round(wmape, 2),
                                                                  nqds,
                                                                  np.round((train_end - train_start).total_seconds(), 2)
                                                                  # Processing Time
                                                                  ], df_result_column_names).T], ignore_index=True)
                        df_res.to_csv(path2main_output_folder + path2csvs + f'{atts_name}_ml_results.csv',
                                      index=False)
                        print(f'\033[92m RMSE {rMSE} MAE {mae} WMAPE {wmape} NQDS {nqds} \033[0m')
                        np.save(
                            path2main_output_folder + path2signal + f'{w}_{atts_name}_{reg}_{lg}_{prd}_{time_cv}.npy',
                            predData)

# This function trains deep learning models on the UNISIM-IV dataset
# Make sure the configuration of GPU is set in DL_Regressors.py 
def train_DL_models():
    from DL_Regressors import call_DL_Regressor
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
    from scikeras.wrappers import KerasRegressor
    
    df_res = pd.DataFrame(columns=df_result_column_names)

    for w_idx, w in enumerate(well_id):
        print(f'**** \033[91m Processing on Well_ID = {w} with input attributes {atts_name} to predict {model_prediction} \033[0m')
        hist_df_main, forecast_df_main = load_data(well_id=w, wtype='P')
        for lg in lag_sizes:
            for prd in periods_h:
                df = pd.concat([hist_df_main, forecast_df_main])
                df = df.loc[:history_ends+prd]
                df_main = data_handling(df[attribute_names], attribute_names, predictor=model_prediction, with_lags=True, t_lags=lg, dah=prd)
                print(df_main.columns)
                values = df_main.values
                values = values.astype('float32')
                features = values[:, :-1]
                labels = values[:, -1]

                # separates train and test
                trainX = features[:-prd, :]
                testX = features[-prd:, :]  # forecast
                train_y = labels[:-prd]
                test_y = labels[-prd:]
                print(trainX.shape,testX.shape)
                
                scaler = MinMaxScaler(feature_range=(0, 1)) 
                trainX = scaler.fit_transform(trainX)
                testX = scaler.transform(testX)
                
                scalerLA = MinMaxScaler(feature_range=(0, 1))
                train_y = scalerLA.fit_transform(train_y.reshape(-1, 1)) 

                trainX = trainX.reshape((trainX.shape[0], 1, trainX.shape[1]))
                testX = testX.reshape((testX.shape[0], 1, testX.shape[1]))

                regs = call_DL_Regressor()
                for time_cv in number_of_splits:
                    for reg in regs.keys():
                        print(f'\033[93m Regressor = {reg}, Lag = {lg}, Period = {prd}, TSCV = {time_cv} \033[0m')
                        if os.path.isfile(path2main_output_folder + path2signal + f'{w}_{atts_name}_{reg}_{lg}_{prd}_{time_cv}.npy'):
                            print('\033[91m ***Existed entry*** \033[0m')
                            df_res = pd.read_csv(path2main_output_folder + path2csvs + f'{atts_name}_dl_results.csv')
                            print(len(df_res))
                            continue
                            
                        oilPred, oilForecast, time_cost = regs[reg](trainX, testX, train_y, num_split=time_cv)
                        
                        oilPred = scalerLA.inverse_transform(oilPred.reshape(-1, 1))
                        oilForecast = scalerLA.inverse_transform(oilForecast.reshape(-1,1))
                        
                        rMSE = np.sqrt(mean_squared_error(test_y[:prd], oilForecast[:prd]))
                        mae = mean_absolute_error(test_y[:prd], oilForecast[:prd])
                        # Calculate the numerator and denominator
                        numerator = np.sum(np.abs(test_y[:prd] - oilForecast[:prd]))
                        denominator = np.sum(np.abs(test_y[:prd]))
                        wmape = numerator / denominator
                        
                        nqds=compute_NQDS(test_y[:prd], oilForecast[:prd])
                        
                        predData = np.vstack((oilPred, oilForecast[:prd]))

                        df_res = pd.concat([df_res, pd.DataFrame([reg,
                                                                  w,
                                                                  lg,
                                                                  prd,
                                                                  time_cv,
                                                                  round(rMSE, 2),
                                                                  round(mae, 2),
                                                                  round(wmape, 2),
                                                                  nqds,
                                                                  round(time_cost.total_seconds(), 2)
                                                                  # Processing Time
                                                                  ], df_result_column_names).T], ignore_index=True)
                        df_res.to_csv(path2main_output_folder + path2csvs + f'{atts_name}_dl_results.csv',
                                      index=False)
                        print("\033[92m RMSE %2.2f, MAE %2.2f, WMAPE %2.2f, NQDS %2.2f \033[0m" % (rMSE, mae, wmape, nqds))
                        np.save(
                            path2main_output_folder + path2signal + f'{w}_{atts_name}_{reg}_{lg}_{prd}_{time_cv}.npy',
                            predData)

                         
# python train.py --model_training ml --model_input OR --model_output OR 
# python train.py --model_training ml --model_input WR --model_output WR 
# python train.py --model_training ml --model_input GR --model_output GR 
if __name__ == '__main__':

    if train_type=='ml':
        #Training standard machine-learning models
        train_ML_models()
    elif train_type=='dl':
        #Training deep-learning models
        train_DL_models()
    #Generate results and forecasting figures consider the eval.py 
    
    
