"""
Author: Bahram Lavi, João Roberto Bertini Junior
Description: This script is used to generate final best models among the ml and dl models considered in the paper:
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
path2signal = 'Signals/'
path2csvs = 'CSVs/'
path2figures = 'Figures/'
path2bestmodels = 'Best_models/'

parser = argparse.ArgumentParser(
        description='A script to get ML model parameters from the command line.'
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
    help="The model prediction output (e.g., 'OR', 'WR', 'GR')."
)
parser.add_argument(
    '--comparison',
    type=str,
    required=True,
    help="To compare the results on which models training among ML and DL only both (to compare only ML pass 'ml', Dl pass 'dl', and both pass 'ml_dl')."
)
# Parse the arguments provided by the user.
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

comparison=args.comparison

atts_name = ''
for i in range(len(attribute_names)):
    atts_name += attribute_names[i] + '_' if i + 1 < len(attribute_names) else attribute_names[i]


# experimental evaluation on the well producers [11...16]
well_id = range(11, 17)
well_state = ['P']
                
history_ends=2891
initial_day_zero_dopping=1200

    
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


 # This function generates the final compariosn of ML and DL model and produce the regression plots of best models through MAE
def generate_results_best_models(model_keys = comparison):
    
    for w_idx,w in enumerate(well_id):
        df = pd.DataFrame()
        if 'ml' in model_keys:
            df_ml = pd.read_csv(path2main_output_folder + path2csvs + f'{atts_name}_ml_results.csv')
            df_ml = df_ml[df_ml.Well == w]
            df = pd.concat((df,df_ml)).reset_index(drop=True)

        if 'dl' in model_keys:
            df_dl = pd.read_csv(path2main_output_folder + path2csvs + f'{atts_name}_dl_results.csv')
            df_dl = df_dl[df_dl.Well==w]
            df = pd.concat((df,df_dl)).reset_index(drop=True)
        
        df_res = df.loc[df.groupby(by=['period'], sort=True)['MAE'].idxmin()]
        for i in range(len(df_res)):
            regressor = df_res.iloc[i].Regressor
            lag = df_res.iloc[i].lag
            period = df_res.iloc[i].period
            tscv = df_res.iloc[i].tscv
            mae, rmse, wmape = df_res.iloc[i].MAE, df_res.iloc[i].RMSE, df_res.iloc[i].WMAPE

            data = np.load(f'./Experiments/Signals/{w}_{atts_name}_{regressor}_{lag}_{period}_{tscv}.npy', allow_pickle=True)                
                
            hist_df_main, forecast_df_main = load_data(well_id=w, wtype='P')
            df = pd.concat([hist_df_main, forecast_df_main])
            df = df.loc[:history_ends+period]
            
            df = data_handling(df, attribute_names, predictor=model_prediction, with_lags=True, impute_zeros=False, t_lags=lag, dah=period)
            
            
            df_final = pd.DataFrame()
            df_final['Day'] = range(initial_day_zero_dopping+period, len(df)+initial_day_zero_dopping+period)
            df_final['Real']=df[model_prediction].values
            df_final['Forecast']= data
            df_final['Forecast']=df_final['Forecast'].shift(period) 

            
            font = {'weight' : 'bold',
                    'size'   : 8}
            plt.rc('font', **font)
            fig, ax1 = plt.subplots(1,3,figsize=(12,3))
            plt.tight_layout(pad=4.5)
            
            ax1[0].plot(df_final.Day[:-period], df_final.Real[:-period],'ro', markersize = 2.5, linewidth=1.5, label='Reference-History')
            ax1[0].plot(df_final.Day[-period:], df_final.Real[-period:],'yo', markersize = 2.5, linewidth=1.5, label='Reference-Forecast')
            ax1[0].plot(df_final.Day[:-period], df_final.Forecast.values[:-period], 'b+', markersize = 3.5, linewidth=1.5, label = 'ML-History')
            ax1[0].plot(df_final.Day[-period:], df_final.Forecast.values[-period:], 'g+', markersize = 3.5, linewidth=1.5, label = 'ML-Forecast')
            ax1[0].legend(loc='lower center', fontsize=6)
            #ax1.set_title(f'P-{w}, {regressor}, period = {period}-day', fontweight='bold', fontsize=10)
            ax1[0].set_ylabel(f'P{w} {model_prediction} \n $[m^3/day]$', fontweight='bold')
            ax1[0].set_xlabel('Time (days)', fontweight='bold')
            #ax1[0].set_xlim(df_final.Day.iloc[0], df_final.Day.iloc[-1])
            ax1[0].grid(axis='y',alpha=0.4, linewidth=1.6)

            
            ax1[1].plot(df_final.Day[:-period], df_final.Real[:-period],'ro', markersize = 5.5, linewidth=.5, label='Reference-History')
            ax1[1].plot(df_final.Day[-period:], df_final.Real[-period:],'yo', markersize = 5.5, linewidth=.5, label='Reference-Forecast')
            ax1[1].plot(df_final.Day[:-period], df_final.Forecast.values[:-period], 'b+', markersize = 5.5, linewidth=1.5, label = 'ML-History')
            ax1[1].plot(df_final.Day[-period:], df_final.Forecast.values[-period:], 'g+', markersize = 5.5, linewidth=1.5, label = 'ML-Forecast')
            ax1[1].legend(loc='lower center', fontsize=8)
            #ax2.set_title(f'Zoom-in, P-{w}, {regressor}, period = {period}-day', fontweight='bold', fontsize=10)
            ax1[1].set_ylabel(f'P{w} {model_prediction} \n $[m^3/day]$', fontweight='bold')
            ax1[1].set_xlabel('Time (days)', fontweight='bold')
            ax1[1].set_xlim(df_final.Day.iloc[-1]-period*2, df_final.Day.iloc[-1])
            ax1[1].set_xticks([history_ends-period, history_ends, history_ends+period])
            ax1[1].grid(axis='y', alpha=0.4, linewidth=1.6)
            ax1[1].legend(loc='lower center', fontsize=6)
            
            df_final2=df_final.iloc[-period:].copy()
            df_final2.drop(index=df_final2[df_final2.Real[-period:]==0].index, axis=0, inplace=True)

            ape = ((df_final2.Real.values - df_final2.Forecast.values) / df_final2.Real.values) * 100
            ax1[2].scatter(df_final2.Day.values, ape, color='blue', alpha=0.6, label="APE at each point")
            ax1[2].axhline(0.0, color='red', linestyle='--')
            #ax1[2].scatter(y=residuals, x=df_final2.Day, s = 5.5, color='b')
            ax1[2].set_xlabel('Time (days)', fontweight='bold')
            ax1[2].set_ylabel('APE (%)', fontweight='bold')
            #ax1[2].set_ylim([-40,40])
            #ax1[2].axhline(y=0, color='r', linestyle='--')
            ax1[2].grid(axis='both',alpha=0.4, linewidth=1.6)
            #ax1[2].legend()

            

            plt.suptitle(f'{period}-Day forecast, {regressor}', fontsize=12, fontweight='bold')
            plt.savefig(f'{path2main_output_folder+path2figures}{atts_name}_p{w}_{regressor}_{period}_{comparison}_Regs.png',dpi=200,bbox_inches='tight')


        df_res = df_res[['Well','period','Regressor', 'lag', 'tscv', 'MAE', 'RMSE','WMAPE', 'NQDS', 'Proc. Time' ]]
        df_res.to_csv(path2main_output_folder+path2bestmodels+f'{attribute_names}_p{w}_{comparison}_best_models.csv', index=False)

                         
if __name__ == '__main__':

    #Generate results and forecasting figures
    generate_results_best_models()

# python eval.py --comparison ml_dl --model_input OR --model_output OR