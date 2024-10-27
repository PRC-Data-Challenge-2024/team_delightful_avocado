# Welcome to the GitHub repository of team delightful avocado 🥑 for the PRC Data Challenge!

This repository contains the code and data for the Performance Review Commission Data Challenge.
More information about the challenge can be found [here](https://ansperformance.eu/study/data-challenge/).

## Installation and Execution
To install the required Python (this repository was made with Python 3.11) packages, run the following command:
```bash
pip install -r requirements.txt
```
The challenge set's combined parquet file is too large to be stored in this repository.
Please run the following command to combine the daily parquet files into a single file:
```bash
cd data
python combine_parquet.py
cd ..
```

You can then run the training script to train the model and generate the submission file:
```bash
python train_model.py
```
or run Bayesian optimisation:
```bash
python bayesian_search.py
```

Due to incombatibilities between SHAP and HistGradientBoostingRegressor, the SHAP model has to be trained separately:
```bash
python train_shap.py
```
Afterward, you can run the SHAP analysis in the notebook:
```bash
jupyter notebook feature_importances.ipynb
```


## Data Preparation
You can find all the used data in the [`data`](data) folder. The data is already preprocessed and ready to be used for training and evaluation.
Used features and sources can be found [README_datapreparation.md](README_datapreparation.md).
The data is computed for the challenge set and for the (final) submission set.
For each set, we can compute a feature matrix X and for the challenge set, we can compute the target vector y (the given takeoff weights).
After all data sources have been prepared the feature matrix X and target vector y can be obtained by calling `get_data()` in [`create_Xy.py`](create_Xy.py).

## Training and Model Selection

In this project, a [HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html) was chosen as it performs exceptionally well on structured, tabular data.

We performed hyperparameter optimisation using Bayesian optimisation with the [scikit-optimize](https://scikit-optimize.github.io/stable/) library.

## Evaluation

We can now take a look at the models parameters and evaluate the inference of each individual parameter to the models output.
The underlying model is a HistGradientBoostingRegressor, which can be analyzed using the `shap` library.
SHAP (SHapley Additive exPlanations) values can be used to explain the model's predictions and understand the importance of each feature in the prediction.
You can find detailed values and plots in the [`feature_importances.ipynb`](feature_importances.ipynb) notebook.

Most important features are `mtom, aircraft_type, oew` which are closely related to the airplane type.
Therefore, we assume the models performance could be improved by adding airplane variants to the dataset.
We also provide some mean values of the feature parameters of the challenge set for the different aircraft types in the [`data/mean_data.csv`](data/mean_data.csv) file.

Further important features are the mean climb rate value during the climb phase having an inverse effect on the takeoff weight, e.g. indicating an empty aircraft having better climb performance.
Interestingly, there also is an influence of the airline indicating maybe different operating specifics.


## Results
The code in this repository, precisely the `train_model.py` script, will reproduce v16 of the submission, giving a root mean square error of 2,355.61kg.


## Authors

Felix Soest @fsoest

Paul Hollmann @paulhollmann
