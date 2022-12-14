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


def scaling():
    df_wine = pd.read_csv(
        "C:\\Users\\Alex\\Dropbox\\Projekte\\Lisa Master\\Code\\winequality-white.csv", sep=";", header=0)
    df_wine.head()  # for more information such as mean and std per feature use .describe()
    df_wine.isnull().sum()
    X = df_wine.drop(columns='quality')
    y = df_wine.quality
    y = y.values
    y = y.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    scaler_features = StandardScaler(copy=True, with_mean=True, with_std=True).fit(X_train)
    scaler_targets = StandardScaler(copy=True, with_mean=True, with_std=True).fit(y_train)
    # sidenote: it is important that we use the same scaling on both, training and test data.
    # use the scaler derived form the training set (as in reality you would have no access to test data yet)
    # Transform the data
    X_train = scaler_features.transform(X_train)
    y_train = scaler_targets.transform(y_train)
    X_test = scaler_features.transform(X_test)
    y_test = scaler_targets.transform(y_test)
    #############################################################################
    # From here on we perform the usual steps
    # Instantiate Regressor
    lin_regr = LinearRegression()
    # Fitting the Regression
    lin_regr.fit(X_train, y_train)
    # Test regressor with test data (and training data to compare them)
    y_pred_test = lin_regr.predict(X_test)
    y_pred_train = lin_regr.predict(X_train)
    # Calc metric (e.g. r2)
    r2_test = r2_score(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    # Get coefficients
    coef = lin_regr.coef_.tolist()[0]


if __name__ == '__main__':
    # np_to_list()
    scaling()
