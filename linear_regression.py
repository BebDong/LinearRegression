# coding=utf-8
# author: BebDong
# date: 2018/11/28
# use only numpy to do linear regression


import numpy as np
from numpy import dot
from numpy.linalg import inv

# read files: 9568 samples, 9569 rows with first row representing the meaning of each column
data_list_matrix = np.loadtxt(open('./sheets/Sheet1.csv'), delimiter=",", skiprows=1)

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

# gradient descent
a_gradient = np.array([1., 1., 1., 1., 1.]).reshape(5, 1)
alpha = 0.00000000025
epochs = 100000
for i in range(epochs):
    error = dot(X_train, a_gradient) - y_train
    error_each_item = dot(X_train.T, error)
    a_gradient = a_gradient - alpha * error_each_item

# prediction
y_prediction = dot(X_test, a)
y_prediction_gradient = dot(X_test, a_gradient)

# mean squared error(MSE) to evaluate
error = y_prediction - y_test
error_gradient = y_prediction_gradient - y_test
MSE = np.sum(pow(error, 2)) / 2392
MSE_gradient = np.sum(pow(error_gradient, 2)) / 2392
print('MSE with least square: ', MSE)
print('MSE with gradient descent: ', MSE_gradient)
