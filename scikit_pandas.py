# coding=utf-8
# author: BebDong
# date: 2018/11/28
# use scikit-learn and pandas to do linear regression


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# use pandas to read files
data = pd.read_csv('./sheets/Sheet1.csv')

# extract attributes and tags of samples from data
X = data[['AT', 'V', 'AP', 'RH']]
y = data[['PE']]

# split the dataset into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# use scikit-learn to do linear regression
linearRegression = LinearRegression()
linearRegression.fit(X_train, y_train)

# print the parameters of the model
print('intercept: ', linearRegression.intercept_)
print('coefficient: ', linearRegression.coef_)

# testing set regression
y_prediction = linearRegression.predict(X_test)

# model evaluation by mean squared error(MSE) or root mean squared error(RMSE)
MSE = metrics.mean_squared_error(y_test, y_prediction)
print('MSE: ', MSE)
