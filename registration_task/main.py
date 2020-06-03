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
    self.w = None
    self.b = None
  
  def fit(self,x,y):
    x = np.array(x)
    m,train_x_shape = x.shape
    self.w = np.zeros(train_x_shape)
    self.b = 0

    for i in range(10000):
      y_pred = np.dot(x, self.w) + self.b
      dw = (1 / m) * np.dot(x.T, (y_pred - y))
      db = (1 / m) * np.sum(y_pred - y)
      self.w =self.w - self.lr * dw
      self.b =self.b - self.lr * db
    
  def predict(self,x):
    y_pred = np.floor((np.dot(x,self.w)+self.b))
    Y = []
    for j in y_pred:
      Y.append(int(j))
    return Y
    
    
