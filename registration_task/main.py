"""
Implement the linear regression model using python and numpy in the following class.
The method fit() should take inputs like,
x = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
"""

import numpy as np

class LinearRegression(object):
  """
  An implementation of linear regression model
  """
  def __init__(self):
    self.lr = 0.001
    self.W = None
    self.b = None
  
  def fit(self,X,y):
    X = np.array(X)
    m,train_X_shape = X.shape
    self.W = np.zeros(train_X_shape)
    self.b = 0

    for _ in range(10000):
      y_pred = np.dot(X, self.W) + self.b
      dw = (1 / m) * np.dot(X.T, (y_pred - y))
      db = (1 / m) * np.sum(y_pred - y)
      self.W =self.W - self.lr * dw
      self.b =self.b - self.lr * db
    
  def predict(self,X):
    y_pred = np.floor((np.dot(X,self.W)+self.b))
    Y = []
    for x in y_pred:
      Y.append(int(x))
    return Y
    
    
