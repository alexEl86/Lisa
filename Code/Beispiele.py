import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


def np_to_list():
    a = np.array(([1, 2, 3], []))
    b = a.tolist()
    c = list(a)
    print(type(b))
    print(type(c))
    print(type(c[0]))
    print()


def scaling(data):
    inputs = data[:,1:]
    outputs = data[:,:1]
    
    #Q: why is transforming first and then splitting the dataset better? why not?
    #A: 
    scaler_input = StandardScaler(copy=True, with_mean=True, with_std=True).fit(inputs)
    scaler_outputs = StandardScaler(copy=True, with_mean=True, with_std=True).fit(outputs)
    scaler_all = StandardScaler(copy=True, with_mean=True, with_std=True).fit(data)
    in_scaled = scaler_input.transform(inputs)
    out_scaled = scaler_outputs.transform(outputs)
    data_scaled = scaler_all.transform(data)
    #print(in_scaled)
    #print(data_scaled)
    #Q: transforming all data is the same as transforming inputs and outputs separately?
    #A: 

    # linear regression
    in_train_scaled, in_test_scaled, out_train_scaled, out_test_scaled = train_test_split(in_scaled, out_scaled, test_size=0.2, random_state=0, shuffle = True)
    _, in_test, _, out_test = train_test_split(inputs, outputs, test_size=0.2, random_state=0, shuffle = True)
    lin_regr = LinearRegression()
    lin_regr.fit(in_train_scaled, out_train_scaled) # -> in scaled / normal space

    # correct prediction and inverse transofrmation
    preds_test_scaled = lin_regr.predict(in_test_scaled)
    preds_test_inv = scaler_outputs.inverse_transform(preds_test_scaled)
    r2_test_scaled = r2_score(out_test_scaled, preds_test_scaled)
    r2_test = r2_score(out_test, preds_test_inv)

    # alternative / wrong procedure
    coef = lin_regr.coef_.tolist()
    coef_inv = scaler_input.inverse_transform(coef)
    lin_regr.coef_ = coef_inv
    preds_test_inv_alt = lin_regr.predict(in_test)
    preds_test_alt = scaler_outputs.inverse_transform(preds_test_inv_alt)
    r2_test_alt = r2_score(out_test, preds_test_alt)

    pass


if __name__ == '__main__':
    # np_to_list()
    data = np.array([[1,2,3],[20,30,40],[3,4,5],[40,50,60],[5,6,7],[60,70,80]])
    scaling(data)
