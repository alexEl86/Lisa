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
    gt = data[:,:1]
    
    #Q: why is transforming first and then splitting the dataset better?
    #A: 
    scaler_input = StandardScaler(copy=True, with_mean=True, with_std=True).fit(inputs)
    scaler_gt = StandardScaler(copy=True, with_mean=True, with_std=True).fit(gt)
    scaler_all = StandardScaler(copy=True, with_mean=True, with_std=True).fit(data)
    inputs_scaled = scaler_input.transform(inputs)
    gt_scaled = scaler_gt.transform(gt)
    data_scaled = scaler_all.transform(data)
    print(inputs_scaled)
    print(data_scaled)
    #Q: transforming all data is the same as transforming inputs and outputs separately?
    #A: 

    inputs_train, inputs_test, gt_train, gt_test = train_test_split(inputs, gt_scaled, test_size=0.2, random_state=0)
    lin_regr = LinearRegression()
    lin_regr.fit(inputs_train, gt_train)
    preds_train = lin_regr.predict(inputs_train)
    preds_test = lin_regr.predict(inputs_test)
    
    coef = lin_regr.coef_.tolist()[0]
    coef_inv = scaler_input.inverse_transform(coef)
    preds_test_inv = scaler_output####

    pass


if __name__ == '__main__':
    # np_to_list()
    data = np.array([[1,2,3],[20,30,40],[3,4,5],[40,50,60],[5,6,7],[60,70,80]])
    scaling(data)
