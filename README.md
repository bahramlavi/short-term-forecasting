This repository provides the code implementation of the paper entitled "Comparing Shallow and Deep Learning Regression Methods to Forecast
Short-Term Oil, Water, and Gas Rates in a Pre-Salt Petroleum Field".

<strong> Case Study: UNISIM-IV</strong>


<strong>Requiements</strong></br>
Make sure your system satisfies with packages requirments specified in requirment.txt file. Or install them by:

<code> pip install requirement.txt</code>

<strong> Training Oil Rate</strong> 
To perform training on the shallow machine learning models, to predict Oil Rate (OR) and consider OR as model input, run the following command:

<code>python train.py --model_training ml --model_input OR --model_output OR</code>

To run the deep learning model pass the <code> --model_training dl</code>

<strong> Training Water Rate</strong> 
To perform training on the shallow machine learning or deep learning models, to predict Water Rate (WR) and consider WR as model input, run the following command:

<code>python train.py --model_training [ml or dl] --model_input WR --model_output WR</code>

<strong> Training Gas Rate</strong> 
To perform training on the shallow machine learning or deep learning models, to predict Gas Rate (GR) and consider GR as model input, run the following command:

<code>python train.py --model_training [ml or dl] --model_input GR --model_output GR</code>

<strong> Additional Model Input Attributes </strong>
To add more attributes to the model input features from the existed attributes in the UNISIM-IV dataset (i.e., OR, WR, GR, BHP), run the following code:

An example on prediction OR with additional attributes [WR, GR, BHP]:

<code>python train.py --model_training [ml or dl] --model_input OR WR GR BHP --model_output OR</code>


Executing the above command will generate a directory named "Experiments", and all the experiments and results will be stored within that folder.

<strong> Model Evaluations </strong>
To generate final results on best models and regression plots, you need to run eval.py with three following input arguments:

<code> --model_input </code>: specifies which model input experiments

<code> --model_out </code>: specifies which model output experiments

<code> --comparision </code>: the comparison is pefromed upon which models (ml models, dl models, or comparing jointly ml and dl models)

to compare ml and dl jointly on [OR] prediction, run:
<code> python eval.py --comparison ml_dl --model_input OR --model_output OR



