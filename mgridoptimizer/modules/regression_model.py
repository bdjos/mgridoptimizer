# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 15:34:33 2018

@author: BJoseph
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import copy

class regression_model:
    def __init__(self, X, y, lr):
        "Initialize supervised regression model with inputs, outputs and learning rate"
        self.X = X
        self.y = y
        self.reg_features = {'X_std': [], 'X_mean': [], 'y_std': None, 'y_mean': None}
        self.X_reg = np.copy(self.X)
        self.y_reg = np.copy(self.X)
        self.learning_rate = lr

    @staticmethod
    def regularize(array, std, mean):
        "Regularize an array using a std deviation and mean input"
        array = copy.copy(array)
#        if len(array) == 1:
#            array = (array-mean)/std
#        else:
#            array = [(x-mean)/std for x in array]
        array = (array-mean)/std
        return array
    
    @staticmethod
    def dereg(array, std, mean):
        "Deregularize an array using a std dev and mean input"
        if len(array) == 1:
            array = array*std+mean
        else:
            array = array*std+mean
        return array
    
    @staticmethod        
    def init_theta(theta_init):
        """Initialize features with numpy array of 1xm features and applies .requires_grad 
        for grad descent"""
        theta = torch.from_numpy(theta_init)
        theta.requires_grad_ ()
        return theta
         
    @staticmethod
    def mse(y_hat, y): 
        return ((y_hat - y) ** 2).mean()
    
    @staticmethod
    def mse_loss(hyp_func, theta, X, y): 
        return regression_model.mse(hyp_func(theta, X), y)
    
    @staticmethod
    def grad_descent(hyp_func, theta, X, y, lr):
        "Run gradient descent using mean squared error on regularized dataset"
        for t in range(10000):
            # Forward pass: compute predicted y using operations on Variables
            loss = regression_model.mse_loss(hyp_func, theta, X, y)
            if t % 1000 == 0: print(loss.item())
            
            # Computes the gradient of loss with respect to all Variables with requires_grad=True.
            loss.backward()
            
            theta.data -= lr * theta.grad.data # Update theta using gradient descent
            theta.grad.data.zero_() # Reset gradients to zero after each loop
        return theta
        
    @staticmethod    
    def standard_grad_descent(cost_func, X, theta, lr, *args):
        "Run gradient descent on a function of inputs X with predetermined theta"
        for t in range (10000):
            loss = cost_func(X, theta, args)
            
            
            loss.backward()
            if t % 1000 == 0: print(X), print(loss.item()),print(X.grad.data)
            X.data -= lr * X.grad.data
            X.grad.data.zero_()
        return X
    
    @staticmethod
    def feature_expand(X):
        "Expand X features. Use only for regression modelling."
        np_temp = torch.from_numpy(np.array(np.ones([X.shape[0], 1])))
        for feature in range(X.shape[1]):
            np_temp = torch.cat((X[:,1-feature, None]**3, X[:,1-feature, None]**2, X[:,1-feature, None], np_temp), 1)
        return np_temp
    
    def deregularize_theta(self):
        """
        Deregularize thetas to predict using non-regularized inputs. Use only for regression modelling
        Factors equation: (y-y_mean)/y_std = theta[0]*((X[0]-X_mean[0])/X_std[0])**3 + ... + theta[6]
        """
        theta = self.theta[0]
        theta_dereg = []
        theta_rems = []
        ys, ym = self.reg_features['y_std'], self.reg_features['y_mean'] # Set vars for cleaner code
        Xs, Xm = self.reg_features['X_std'], self.reg_features['X_mean'] # Set vars for cleaner code
        for i in range(self.X.size()[1]): # Loop through all features
            theta_dereg.append(((theta[(i*3)+0]*ys)/Xs[i]**3).item()) # X**3
            theta_dereg.append(((theta[(i*3)+1]*ys/Xs[i]**2)-(3*Xm[i]*theta[(i*3)+0]*ys/Xs[i]**3)).item()) # X**2
            theta_dereg.append(((theta[(i*3)+2]*ys/Xs[i])-(2*theta[(i*3)+1]*ys*Xm[i]/Xs[i]**2)+(3*(Xm[i]**2)*theta[(i*3)+0]*ys/(Xs[i]**3))).item()) # X
            theta_rems.append(((theta[(i*3)+1]*ys*(Xm[i]**2)/Xs[i]**2)-(theta[(i*3)+0]*ys*(Xm[i]**2)/(Xs[i])**3)-(theta[(i*3)+2]*ys*Xm[i]/Xs[i])).item()) # Remaining
        theta_rems.append((theta[6]*ys + ym).item()) # Add remaining theta and y_mean
        theta_dereg.append(sum(theta_rems))
        theta_dereg = torch.from_numpy(np.array([theta_dereg]))
        return theta_dereg
    
    def torch_inputs(self):
        "Convert inputs and outputs to torch tensor"
        self.X, self.y,  = torch.from_numpy(self.X), torch.from_numpy(self.y)
        self.X_reg, self.y_reg = torch.from_numpy(self.X_reg), torch.from_numpy(self.y_reg)                                         
    
    def init_reg_vars_torch(self):
        "Regularize inputs and outputs of model and save regularization features"
        # Regularize y features
        self.reg_features['y_std'] = self.y.max().item() - self.y.min().item()
        self.reg_features['y_mean'] = self.y.mean().item()
        self.y_reg = torch.from_numpy(np.array(self.regularize(self.y, self.reg_features['y_std'], self.reg_features['y_mean'])))
        
        # Regularize X features
        m, n = self.X.shape
        for feature in range(n):   
            self.reg_features['X_std'].append(self.X_reg[:, feature].max().item() - self.X_reg[:, feature].min().item())
            self.reg_features['X_mean'].append(self.X_reg[:, feature].mean().item())
    
            self.X_reg[:,feature] = torch.from_numpy(np.array(self.regularize(self.X_reg[:,feature], self.reg_features['X_std'][feature], self.reg_features['X_mean'][feature])))
    
    def init_reg_vars(self):
        "Regularize inputs and outputs of model and save regularization features"
        # Regularize y features
        self.reg_features['y_std'] = self.y.max() - self.y.min()
        self.reg_features['y_mean'] = self.y.mean()
        self.y_reg = np.array(self.regularize(self.y, self.reg_features['y_std'], self.reg_features['y_mean']))
        
        # Regularize X features
        m, n = self.X.shape
        for feature in range(n):   
            self.reg_features['X_std'].append(self.X_reg[:, feature].max() - self.X_reg[:, feature].min())
            self.reg_features['X_mean'].append(self.X_reg[:, feature].mean())
    
            self.X_reg[:,feature] = np.array(self.regularize(self.X_reg[:,feature], self.reg_features['X_std'][feature], self.reg_features['X_mean'][feature]))

    
if __name__ == '__main__':        
    
    # Inputs for PV Watts API
    solar_cost = 1.5*1000 # $/kW cost of system
    storage_cost = 1*1000
    energy_cost = 0.25/1000 # $/kWh cost of electricity
    file = os.path.join('data', 'dc_foods_2014.csv')
    min_pv_size = 10
    max_pv_size = 5000
    min_storage_size = 10
    max_storage_size = 5000
    num_steps = 5 # min 3
    
    # Test values for sample of inputs and outputs
    solar = [1.0, 200.0, 350.0, 500.0, 650.0, 800.0]   
    storage = [1.0, 100.0, 200.0, 300.0, 400.0, 500.0]
    
    X = np.array([solar,storage])
    X = X.transpose()
    y = np.array([3259564736.17,
               2968338374.102,
               2775258723.83,
               2603167289.5210004,
               2453123575.3079996,
               2329963297.0230002])
    
    lr = 1e-1
    num_features = 7
    
    def hypothesis(theta, X):
       X = regression_model.feature_expand(X)
       return torch.mm(X, theta.transpose(0,1))[:,0]
   
    model = regression_model(X, y, lr) # Create regression model object
    model.torch_inputs() # Convert numpy inputs to torch tensors
    model.init_reg_vars() # Regularize inputs and store regularization variables 
    model.theta = model.init_theta(np.array([[1.0,2.0,3.0,4.0,5.0,6.0,7.0]])) # Initialize the features with input = number of features
    model.theta = model.grad_descent(hypothesis, model.theta, model.X_reg, model.y_reg, lr) # Run gradient descent to get features
    theta_dereg = model.deregularize_theta()
    
    y_test_dereg = hypothesis(theta_dereg, model.X)
    y_test = hypothesis(model.theta, model.X_reg)
    y_test = model.dereg(y_test, model.reg_features['y_std'], model.reg_features['y_mean'])
    