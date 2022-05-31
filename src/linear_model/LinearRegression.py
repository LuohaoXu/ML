# -*- coding: utf-8 -*-
"""
Created on Wed May 18 23:40:04 2022

@author: Luohao Xu
"""
import numpy as np
import sklearn.linear_model as sk
from sklearn.datasets import fetch_california_housing
# 线性回归
class LinearRegression:
    
    def __init__(self):
        pass
    
    def fit(self,X,y):
        m = len(y) # 样本总数
        X = np.c_[np.matrix(X),np.ones(m)]
        y = np.matrix(y).T
        self.w = np.linalg.pinv(X.T*X)*X.T*y
        
    def predict(self, X):
        X = np.c_[np.matrix(X), np.ones(len(X))]
        predict_y = X * self.w
        return np.array(predict_y.T.tolist()[0])
        
if __name__ == "__main__":
    data = fetch_california_housing()
    X = data.data
    y = data.target

    my_lr = LinearRegression()
    my_lr.fit(X, y)
    res0 = my_lr.predict(X)
    
    sk_lr = sk.LinearRegression()
    sk_lr.fit(X, y)
    res1 = sk_lr.predict(X)
    
    # 比较sklearn和自己实现的结果
    print(res0)
    print(res1)

    