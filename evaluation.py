# coding=utf-8
# author: BebDong
# date: 2018/11/28
# to packaged into functions and do evaluation on these methods


import numpy as np
from numpy import dot
from numpy.linalg import inv

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn import metrics

import matplotlib.pyplot as plt

import time


def visualMSE(y_test, y_prediction, title):

    # visualisation
    figure, ax = plt.subplots()
    ax.scatter(y_test, y_prediction)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    ax.set_xlabel('measured')
    ax.set_ylabel('predicted')
    ax.set_title(title)
    plt.show()

    # MSE
    MSE = metrics.mean_squared_error(y_test, y_prediction)
    return MSE


def least_square():
    # read files: 9568 samples, 9569 rows with first row representing the meaning of each column
    data_list_matrix = np.loadtxt(open('./sheets/Sheet3.csv'), delimiter=",", skiprows=1)

    # extract attributes and tags respectively
    data_list_attribute = data_list_matrix[:, 0:4]
    data_list_tag = data_list_matrix[:, 4:5]

    # we simply split the dataset: top 75% rows as training data(7176) and the rest as testing data(2392)
    X_train = data_list_attribute[:7176]
    X_test = data_list_attribute[7176:]
    y_train = data_list_tag[:7176]
    y_test = data_list_tag[7176:]

    # add a column to X_train filled with 1
    X_train = np.c_[X_train, np.ones((7176, 1))]
    X_test = np.c_[X_test, np.ones((2392, 1))]

    # least square method: w = ((X'X)^-1)X'y
    a = dot(dot(inv(dot(X_train.T, X_train)), X_train.T), y_train)
    # prediction
    y_prediction = dot(X_test, a)
    MSE_least_square = visualMSE(y_test, y_prediction, 'least square without sklearn')
    print('MSE when least square: ', MSE_least_square)

    # gradient descent
    a_gradient = np.array([1., 1., 1., 1., 1.]).reshape(5, 1)
    alpha = 0.00000000025
    epochs = 100000
    for i in range(epochs):
        error = dot(X_train, a_gradient) - y_train
        error_each_item = dot(X_train.T, error)
        a_gradient = a_gradient - alpha * error_each_item
    # prediction
    y_prediction = dot(X_test, a_gradient)
    MSE_gradient = visualMSE(y_test, y_prediction, 'gradient descent')
    print('MSE when gradient descent: ', MSE_gradient)


def sklearn_way():
    # use pandas to read files
    data = pd.read_csv('./sheets/Sheet3.csv')

    # extract attributes and tags of samples from data
    X = data[['AT', 'V', 'AP', 'RH']]
    y = data[['PE']]

    # split the dataset into training set and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    # use scikit-learn to do linear regression
    linearRegression = LinearRegression()
    linearRegression.fit(X_train, y_train)
    # testing set regression
    y_prediction = linearRegression.predict(X_test)
    # MSE
    MSE = visualMSE(y_test, y_prediction, 'least square with sklearn')
    print('MSE when sklearn: ', MSE)

    # 10 cross validation
    prediction = cross_val_predict(linearRegression, X, y, cv=10)
    MSE_cross_vali = visualMSE(y, prediction, 'MSE when cross validation')
    print('MSE when cross validation: ', MSE_cross_vali)


def main():
    # begin
    start = time.perf_counter()
    # linear regression
    least_square()
    sklearn_way()
    # end
    end = time.perf_counter()
    print("time cost: ", end - start)


if __name__ == '__main__':
    main()
