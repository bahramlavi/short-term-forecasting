This repository provides the code implementation of the paper entitled "Comparing Shallow and Deep Learning Regression Methods to Forecast
Short-Term Oil, Water, and Gas Rates in a Pre-Salt Petroleum Field".

<strong> Case Study: UNISIM-IV</strong>

This work focuses on the UNISIM-IV benchmark dataset as a case study to evaluate shallow machine learning and deep learning approaches on short-term production rate forecasts. UNISIM-IV is an open-source benchmark case based on a light-oil carbonate Brazilian pre-salt field with a high oil-gas ratio and CO2 content. In the production strategy, the database consisted of six vertical producers (which are indexed as (P11 to P16) and seven vertical WAG-CO2 injectors. 
This study only focuses on the producer's data. Each producer’s data contains the daily information on Oil Rate (OR), Water Rate (WR), and Gas Rate (GR).

The producers' dataset (P11 to P16) can be found in the './U4_Data/' folder. 

<strong>Requiements</strong></br>
Make sure your system satisfies with package requirements specified in the requirements.txt file. Or install them by:

<code> pip install requirements.txt</code>

<strong> Training Oil Rate</strong> 
To perform training on the shallow machine learning models, to predict Oil Rate (OR) and consider OR as model input, run the following command:

<code>python train.py --model_training ml --model_input OR --model_output OR</code>

To run the deep learning model pass the <code> --model_training dl</code>

The GPU device name (if any exists) can be configured within the DL_Regressors.py file. 

<strong> Training Water Rate</strong> 
To perform training on the shallow machine learning or deep learning models, to predict Water Rate (WR) and consider WR as model input, run the following command:

<code>python train.py --model_training [ml or dl] --model_input WR --model_output WR</code>

<strong> Training Gas Rate</strong> 
To perform training on the shallow machine learning or deep learning models, to predict Gas Rate (GR) and consider GR as model input, run the following command:

<code>python train.py --model_training [ml or dl] --model_input GR --model_output GR</code>

<strong> Additional Model Input Attributes </strong>
To add more attributes to the model input features from the existing attributes in the UNISIM-IV dataset (i.e., OR, WR, GR, BHP), run the following code:

An example of predicting OR with additional attributes [WR, GR, BHP]:

<code>python train.py --model_training [ml or dl] --model_input OR WR GR BHP --model_output OR</code>


Executing the above command will generate a directory named "Experiments", and all the experiments and results will be stored within that folder.

<strong> Model Evaluations </strong>
To generate final results on best models and regression plots, you need to run eval.py with the following three input arguments:

<code> --model_input </code>: specifies which model input experiments

<code> --model_out </code>: specifies which model output experiments

<code> --comparison </code>: the comparison is performed upon which models (ml models, dl models, or comparing jointly ml and dl models)

to compare ml and dl jointly on [OR] prediction, run:

<code> python eval.py --comparison ml_dl --model_input OR --model_output OR </code>



