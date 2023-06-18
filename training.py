import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, r_regression
from pathlib import Path

import data_prep

data_preper = data_prep.Dataprep("well_annotated_drugs.csv",
                                 "nci60_logIC50s_small.csv",
                                 "nci60_tpm.tsv")

# tpm_T_np, logIC50_740_T_np = data_preper.convert_data()
tpm_T, logIC50 = data_preper.convert_data()

Losses = []
coefficients = []

for drug in data_preper.drugs["NSC"]:

    logIC50_drug = logIC50[logIC50["NSC"] == drug]

    # extract cell lines tested with drug
    tpm_T_np = data_preper.extract_cell_lines(tpm_T, logIC50_drug)

    logIC50_drug.drop(["NSC"], inplace=True, axis=1)
    logIC50_drug_np = logIC50_drug.to_numpy()

    '''
    # first attempt using only one gen as a feature
    tpm_T_np_1_gen = tpm_T_np[:, [0]]
    '''

    # feature selection using r_regression
    n_features = int(np.floor(tpm_T_np.shape[1] /100 * 10)) # keep 10% of features
    # r_regression(tpm_T_np, logIC50_740_np) # calculates pearsons R for each feature in relation to Tpm
    tpm_T_np_new = SelectKBest(r_regression, k=n_features).fit_transform(tpm_T_np, logIC50_drug_np)

    # Split the data into training/testing sets
    tpm_T_np_new_train, tpm_T_np_new_test, logIC50_drug_np_train,\
        logIC50_drug_np_test = data_preper.split_data(tpm_T_np_new, logIC50_drug_np)

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(tpm_T_np_new_train, logIC50_drug_np_train)

    # Make predictions using the testing set
    logIC50_drug_np_pred = regr.predict(tpm_T_np_new_test)

    # calculate loss
    MSE = mean_squared_error(logIC50_drug_np_test, logIC50_drug_np_pred)

    # append loss and coefficients of the model
    Losses.append(MSE)
    coefficients.append(regr.coef_)

list(data_preper.drugs["DRUG_NAME"])

data = {'Drug': list(data_preper.drugs["DRUG_NAME"]),
        'Loss': Losses,
        'coefficients': coefficients}

results = pd.DataFrame(data)
results.set_index(["Drug"], inplace=True)

results.to_csv(Path("/Users/michaelbaggiolorenz/Desktop/drug response/drug_models.csv"))
