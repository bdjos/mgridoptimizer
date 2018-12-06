# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 15:34:33 2018

@author: BJoseph
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

class regression_model:
    def __init__(self, X, y, lr):
        self.X = X
        self.y = y
        self.learning_rate = lr

    @staticmethod
    def regularize(array, std, mean):
        if len(array) == 1:
            array = (array-mean)/std
        else:
            array = [(x-mean)/std for x in array]
        return array
        
    @staticmethod
    def lin(a, b, c, d, x):
        return a*x**3+b*x**2+c*x+d
    
    @staticmethod
    def dereg(array, std, mean):
        if len(array) == 1:
            array = array*std+mean
        else:
            array = [x*std+mean for x in array]
        return array
    
    def torch_inputs(self):
        self.X, self.y = torch.from_numpy(self.X), torch.from_numpy(self.y)
    
    def init_reg_vars(self):
        self.X_std = self.X.max() - self.X.min()
        self.y_std = self.y.max() - self.y.min()
        self.X_mean, self.y_mean = self.X.mean(), self.y.mean()
        self.X_reg = torch.from_numpy(np.array(self.regularize(self.X, self.X_std, self.X_mean)))
        self.y_reg = torch.from_numpy(np.array(self.regularize(self.y, self.y_std, self.y_mean)))
    
    def init_theta(self):
        self.a = torch.from_numpy(np.random.randn(1))
        self.b = torch.from_numpy(np.random.randn(1))
        self.c = torch.from_numpy(np.random.randn(1))
        self.d = torch.from_numpy(np.random.randn(1))
        self.a.requires_grad_ ()
        self.b.requires_grad_ ()
        self.c.requires_grad_ ()
        self.d.requires_grad_ ()
    
    def mse(self, y_hat, y): 
        return ((y_hat - y) ** 2).mean()
    
    def mse_loss(self, a, b, c, d, x, y): 
        return self.mse(self.lin(a,b,c,d,x), y)
    
    def grad_descent(self):
        for t in range(10000):
            # Forward pass: compute predicted y using operations on Variables
            loss = self.mse_loss(self.a,self.b,self.c,self.d,self.X_reg,self.y_reg)
            if t % 1000 == 0: print(loss.item())
            
            # Computes the gradient of loss with respect to all Variables with requires_grad=True.
            # After this call a.grad and b.grad will be Variables holding the gradient
            # of the loss with respect to a and b respectively
            loss.backward()
            
            # Update a and b using gradient descent; a.data and b.data are Tensors,
            # a.grad and b.grad are Variables and a.grad.data and b.grad.data are Tensors
            self.a.data -= self.learning_rate * self.a.grad.data
            self.b.data -= self.learning_rate * self.b.grad.data
            self.c.data -= self.learning_rate * self.c.grad.data
            self.d.data -= self.learning_rate * self.d.grad.data
            
            # Zero the gradients
            self.a.grad.data.zero_()
            self.b.grad.data.zero_()    
            self.c.grad.data.zero_() 
            self.d.grad.data.zero_() 
        
        return self.a, self.b, self.c, self.d
    
    def standard_grad_descent(self, cost_func, theta, lr):
        for t in range (10000):
            loss = cost_func(self.a.item(), self.b.item(), self.c.item(), self.d.item(), self.X_std, self.X_mean, self.y_std, self.y_mean, theta)
            if t % 1000 == 0: print(loss.item())
            
            loss.backward()
            
            theta.data -= lr * theta.grad.data
            theta.grad.data.zero_()
        return theta
    
if __name__ == '__main__':        
        
    sizes = np.array([1.0, 200.0, 350.0, 500.0, 650.0, 800.0])
    sizes_reg, sizes_std, sizes_mean = regression_model.regularize(sizes)
    demands = np.array([3259564736.17,
               2968338374.102,
               2775258723.83,
               2603167289.5210004,
               2453123575.3079996,
               2329963297.0230002])
    demands_reg, demands_std, demands_mean = regression_model.regularize(demands)
    
    lr = 1e-1
    
    reg_model = regression_model.torch_inputs(sizes_reg, demands_reg, lr)
    reg_model.init_theta()
    a, b, c, d = reg_model.grad_descent()
    
    # Evaluate one result
    y_test = regression_model.lin(a, b, c, d, x=(200-sizes_mean)/sizes_std)
    
    # Deregularize results
    y_test = (y_test*demands_std)+demands_mean
    
    # Evaluate and deregularize initial test values
    y_init = regression_model.lin(a, b, c, d, x=torch.from_numpy(sizes_reg))
    y_hat = regression_model.dereg(y_init, demands_std, demands_mean)
    
    plt.plot(sizes, y_hat)
    plt.plot(sizes, demands)
    plt.show()
    
    
    