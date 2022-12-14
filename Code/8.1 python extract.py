# %% [markdown]
# # Exercise 8 - Model interpretation
#
# This exercise focuses on model interpretation. In this context model interpretation means the contributions of individual features to the prediction outcome.
#
# #### Contact
#
# As always if you have any questions regarding the notebook or the course, please contact:
#
# David Steyrl: david.steyrl@univie.ac.at
#
# And please ask questions in the forum so we can answer them directly or pick them up in a Q&A :)
#
# Some helpful resources (accessed 21.11.2022):
#
# - https://machinelearningmastery.com/calculate-feature-importance-with-python/
#
# - https://realpython.com/linear-regression-in-python/
#
# - https://pythonprogramming.net/regression-introduction-machine-learning-tutorial/
#
# - https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/
#
# - https://christophm.github.io/interpretable-ml-book/

# %%
# Import packages that we will need in the exercise
from sklearn.datasets import fetch_california_housing
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold, train_test_split

# %% [markdown]
# ## Preparing the data
#
# We use the wine quality data set in this exercise. As usual, we try to predict the wine quality from a battery of wine metrics, e.g. residual sugar, citric acid, etc. This time, however, the final goal is to understand which of the features are important to the prediction outcome.

# %%
# Import wine quality data from .csv file
df_wine = pd.read_csv("C:\\Users\\Alex\\Dropbox\\Projekte\\Lisa Master\\Code\\winequality-white.csv", sep=";", header=0)
df_wine.head()  # for more information such as mean and std per feature use .describe()

# %%
# Check for missing values
df_wine.isnull().sum()

# %%
# Preparing data assigning features and target
X = df_wine.drop(columns='quality')
y = df_wine.quality
y = y.values.reshape(-1, 1)

# %% [markdown]
# Since we workd through this already we keep the prediction part simple using a train-test split.
# However, make sure you understand the code and consult the previous exercises or post to the forum if something is not clear! :)

# %%
# 1. Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# Instantiate Regressor
lin_regr = LinearRegression()
# 2. Fitting the Regression
lin_regr.fit(X_train, y_train)
# 3. Test regressor with test data (and training data to compare them)
y_pred_test = lin_regr.predict(X_test)
y_pred_train = lin_regr.predict(X_train)
# 4. Calc metric (e.g. r2)
r2_test = r2_score(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
# We now have our predictions and the overall performance of our model
print('Predictions: ', y_pred_test)

print('Train performance: ', r2_train)

print('Test performance: ', r2_test)

# %% [markdown]
# As expected the performance on the test set is lower than the performance on the training set.

# %% [markdown]
# # Model Inspection
#
# ## Linear Regression
#
# Remember: Linear Regression fits a linear model with coefficients w = (w1, …, wp) (w ist das gleiche wie b) to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.
#
# We can access the coeffiecients using .coef_ on our Linear Regression object.

# %%
# What are our coefficients?
coef = lin_regr.coef_
print(coef)

# %%
# The coefficients are returned as a numpy array. We will convert the array to a list to plot our coefficients.
print(type(coef))
coef = coef.tolist()[0]  # we can do so using .tolist()
# we use [0] to get the first element of our list, to make sure we have a single list instead of lists of lists
# feel free to play around with the [0] if you are curious about it
print(type(coef))
print(coef)

# %%
# Get labels for the features as a list.
# Returns list of features except the target (quality) --> [:-1]
labels = df_wine.columns.values[:-1].tolist()
print(labels)

# %%
# Plot coefficients for each feature
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
plt.xticks(rotation=90)
ax.bar(labels, coef)
plt.xlabel('Features')
plt.ylabel('coefficient values')
plt.show()

# %% [markdown]
# Let's have a look at our coefficients. Which one is the most important one?
# Well, at this point we cannot really tell. The problem is the different scales of our features. Let's have another look at the values of our data.

# %%
X.head()

# %% [markdown]
# Linear machine learning algorithms fit a model where the prediction is the weighted sum of the input values .
#
# These algorithms find a set of coefficients to use in the weighted sum in order to make a prediction. These coefficients can be used directly as a feature importance score. However, there is a problem. The weights depend on the scale of the imput variable. The weights are small if the values of a variable are big and vice versa. One way to cope with this problem is standardization.
#
# Short recap on standardization:
#
# Standardization is the process of putting different variables on the same scale. This process allows you to compare scores between different types of variables. Typically, to standardize variables, you calculate the mean and standard deviation for a variable. Then, for each observed value of the variable, you subtract the mean and divide by the standard deviation.
#
# In Scikit learn we can use the StandardScaler to apply standardization.

# %%
# Import standardizer package
# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
#############################################################################
# Here is the StandardScaler application
# Standardize data using StandardScaler -
# Create and fit StandardScaler
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

# %%
# Our model performs equally well
print(r2_train)
print(r2_test)

# %%
# Let's look at our coefficients
print(coef)

# %%
# Create the same plot as before
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
plt.xticks(rotation=90)
ax.bar(labels, coef)
plt.xlabel('Features')
plt.ylabel('coefficient values')
plt.show()

# %% [markdown]
# Now that the features are on the same scale we can compare the coefficients.
#
# However, since we standardized them, the usual interpretation of the weights is no longer true. Usually, the coefficients indicate the increase/decrease along the y-axis with each unit of increase along the x-axis.
# To be able to interpret the coefficients that way again we have to inverse the applied transform.
#

# %%
# inverse transform the coefficients
coef_retransformed = scaler_features.inverse_transform(lin_regr.coef_)
# we are using the coef np.array instead of the coef list as the method expects that type
print(coef_retransformed)

# %% [markdown]
# To sum up: We have to standardize our features before we put them into our linear regression model to allow us to compare their importance to one another.
# In order to interpret their effect on our target variable in the usual way we inverse the applied transform to get the weight back in the scale of the original data.

# %% [markdown]
# ## Decision Trees
#
# When we inspect a linear regression model we take a closer look at the coefficients to gain an understanding about our features and their importance.
# In tree based methods there are no coefficients as in linear regression. However, we can inspect the importance of our features as measured by their contribution to the error reduction.
# The importance of a feature is computed as the (normalized) total reduction of the criterion brought by that feature, e.g. how much is the MSE reduced by a specific feature.

# %%
# Note: We would not have to redo the standarization since the data is still standardized.
# However, we left it in so you can switch between standardized/non-standardized.
# Go ahead and try it both ways and observe what happens!
# Preparing data assigning features and target - prepared for reloading data / exploring standardization
X = df_wine.drop(columns='quality')
y = df_wine.quality.values.reshape(-1, 1)
# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# Create and fit StandardScaler
scaler_features = StandardScaler(copy=True, with_mean=True, with_std=True).fit(X_train)
# Transform the data
X_train = scaler_features.transform(X_train)
X_test = scaler_features.transform(X_test)
# Instantiate Regressor, using max_depth=6, since we already found out in exercise 6 that it is the best setting
dec_tree = DecisionTreeRegressor(max_depth=6)
# Fitting the Decision tree
dec_tree.fit(X_train, y_train)
# Test DT with test data (and training data to compare them)
y_pred_test = dec_tree.predict(X_test)
y_pred_train = dec_tree.predict(X_train)
# Calc metric (e.g. r2)
r2_test = r2_score(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)

# %%
print(r2_train)
print(r2_test)

# %%
'''
The importance of features is automatically computed during the training of the DT and 
stored in the feature_importance_ variable
'''
print(dec_tree.feature_importances_)

# %%
# Convert the features importance to list
feat_imp = dec_tree.feature_importances_.tolist()
print(feat_imp)

# %%
# Create the same plot as for the feature importance in the linear model
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
plt.xticks(rotation=90)
ax.bar(labels, feat_imp)
plt.xlabel('Features')
plt.ylabel('Feature Importances - Decision Tree')
plt.show()

# %% [markdown]
# Here we see the contributions of each feature to the overall MSE reduction of the decision tree. The results are not the same as for the linear regression. Features have other importance. This is reasonable, since the performance of the two methods is not the same as well.
#
# One more note is standardization in tree based methods:
# Tree-based algorithms are fairly insensitive to the scale of the features. Think about it, a decision tree is only splitting a node based on a single feature. The decision tree splits a node on a feature that increases the homogeneity of the node/reduces the error. This split on a feature is not influenced by other features.
#
# Explore this behaviour in the example above by commenting/uncommenting the lines of code that implements the scaling! :)
#
# Let's continue with the the random forest which will look similar.

# %% [markdown]
# ## Random Forest

# %%
# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# Create and fit StandardScaler
scaler_features = StandardScaler(copy=True, with_mean=True, with_std=True).fit(X_train)
scaler_targets = StandardScaler(copy=True, with_mean=True, with_std=True).fit(y_train)
# Transform the data
X_train = scaler_features.transform(X_train)
y_train = np.ravel(scaler_targets.transform(y_train))
X_test = scaler_features.transform(X_test)
y_test = np.ravel(scaler_targets.transform(y_test))
# Instantiate Regressor, restricting n_estimators only to reduce computation time
rf_regr = RandomForestRegressor(n_estimators=50)
# Fitting the RF
rf_regr.fit(X_train, y_train)
# Test RF with test data (and training data to compare them)
y_pred_test = rf_regr.predict(X_test)
y_pred_train = rf_regr.predict(X_train)
# Calc metric (e.g. r2)
r2_test = r2_score(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)

# %%
print(r2_train)
print(r2_test)

# %%
"""
The importance of features is automatically computed during the training of the DT and 
stored in the feature_importance_ variable
"""
rf_feat_imp = rf_regr.feature_importances_.tolist()
print(rf_feat_imp)

# %%
# Create the same plot as before
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
plt.xticks(rotation=90)
ax.bar(labels, rf_feat_imp)
plt.xlabel('Features')
plt.ylabel('Feature Importances -  Random Forest')
plt.show()

# %% [markdown]
# # Congrats!
#
# You worked through another session that expands your knowledge from last week. You now know how to evaluate a model regarding which feature are important for it's predictions.

# %% [markdown]
# # Exercise
#
# Now we are going to look at the California housing dataset again. We already know how to predict the prices but don't know yet what the important variables are for doing so.
#
# Feature importance analyses is also very important in psychological research since feature importance analysis enables you to unravel the underlying mechanisms behind a phenomenon a bit and enables you to design better experiemnts helping you to focus on the important aspects.
#
# Task:
# Import the California housing set and use only the first 1000 rows. Step through the usual steps of data preparation and model training and testing. You do not need to worry about cross-validation (use a train-test split as we did in this exercises) and hyperparameter tuning this time. Instead take the time and explore feature importances using Linear Regression, Decision Trees and Random Forests.

# %%
# Import data
california_housing = fetch_california_housing(as_frame=True)


# %%
california_df = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)

california_df['PRICE'] = pd.Series(california_housing.target)

california_df = california_df.iloc[0:1000]

california_df.head()

# %%
X = california_df.drop(columns='PRICE')
y = california_df.PRICE
# print(y)
y = y.values.reshape(-1, 1)
# print(y)

# %%
# 1. Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# Instantiate Regressor
lin_regr = LinearRegression()
# 2. Fitting the Regression
lin_regr.fit(X_train, y_train)
# 3. Test regressor with test data (and training data to compare them)
y_pred_test = lin_regr.predict(X_test)
y_pred_train = lin_regr.predict(X_train)
# 4. Calc metric (e.g. r2)
r2_test = r2_score(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
# We now have our predictions and the overall performance of our model
print('Predictions: ', y_pred_test)

print('Train performance: ', r2_train)

print('Test performance: ', r2_test)

# %%
coef = lin_regr.coef_  # Q: Warum macht man das?
print(coef)
print(type(coef))

coef = coef.tolist()[0]
print(coef)
print(type(coef))


# %%
labels = california_df.columns.values[:-1].tolist()
print(labels)

# %%
# Import standardizer package
# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
#############################################################################
# Here is the StandardScaler application
# Standardize data using StandardScaler -
# Create and fit StandardScaler
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


# %%
print(r2_train)
print(r2_test)

# %%
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
plt.xticks(rotation=90)
ax.bar(labels, coef)
plt.xlabel('Features')
plt.ylabel('coefficient values')
plt.show()


# %%
# Note: We would not have to redo the standarization since the data is still standardized.
# However, we left it in so you can switch between standardized/non-standardized.
# Go ahead and try it both ways and observe what happens!
# Preparing data assigning features and target - prepared for reloading data / exploring standardization
#X = df_wine.drop(columns='quality')
#y = df_wine.quality.values.reshape(-1, 1)
# Split data into training and test set
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# Create and fit StandardScaler
#scaler_features = StandardScaler(copy=True, with_mean=True, with_std=True).fit(X_train)
# Transform the data
##X_train = scaler_features.transform(X_train)
#X_test = scaler_features.transform(X_test)
# Instantiate Regressor, using max_depth=6, since we already found out in exercise 6 that it is the best setting
dec_tree = DecisionTreeRegressor(max_depth=6)
# Fitting the Decision tree
dec_tree.fit(X_train, y_train)
# Test DT with test data (and training data to compare them)
y_pred_test = dec_tree.predict(X_test)
y_pred_train = dec_tree.predict(X_train)
# Calc metric (e.g. r2)
r2_test = r2_score(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)

print(r2_train)
print(r2_test)

'''
The importance of features is automatically computed during the training of the DT and 
stored in the feature_importance_ variable
'''
print(dec_tree.feature_importances_)

# Convert the features importance to list
feat_imp = dec_tree.feature_importances_.tolist()
print(feat_imp)

# Create the same plot as for the feature importance in the linear model
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
plt.xticks(rotation=90)
ax.bar(labels, feat_imp)
plt.xlabel('Features')
plt.ylabel('Feature Importances - Decision Tree')
plt.show()

# %%
#preds_x_test = lin_regr.predict(X_test)
# print(preds_x_test)

# inverse transform the coefficients
coef_retransformed = scaler_features.inverse_transform(lin_regr.coef_)
# we are using the coef np.array instead of the coef list as the method expects that type
# print(coef_retransformed)
lin_regr.coef_ = coef_retransformed
preds_x_test = lin_regr.predict(X_test)
# print(preds_x_test)


# %%


# %%
