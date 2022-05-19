# -*- coding: utf-8 -*-
"""
Created on Wed May 18 23:40:04 2022

@author: Luohao Xu
"""
import numpy as np

# 线性回归
class LinearRegression:
    
    def __init__(self, X, y):
        self.m = len(y) # 样本总数
        self.d = len(X[0]) # 特征总数
        self.X = np.c_[np.matrix(X),np.ones(self.m)]
        self.y = np.matrix(y).T
        self.fit()
    
    def fit(self):
        X, y = self.X, self.y
        self.w = np.linalg.pinv(X.T*X)*X.T*y
        
    def predict(self, predict_X):
        X = np.c_[np.matrix(predict_X), np.ones(len(predict_X))]
        predict_y = X * self.w
        return predict_y
        
if __name__ == "__main__":
    X = [[1],[2],[3],[4]]
    y = [5,9,13,17]
    lr = LinearRegression(X, y)
    result = lr.predict([[5],[6]])
    print(result)
    