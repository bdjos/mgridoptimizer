# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 15:29:25 2018

@author: BJoseph
"""

import sys
sys.path.append('/models')

import os
import mgridtest
from modules.mgrid_model import *
from modules.regression_model import *
from modules.nn_model import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from itertools import product
import copy


lr = 1e-3 # Set learning rate

# Define all input ranges
solar_min = 10
solar_max = 1000
battery_min = 10
battery_max = 1000
converter_min = 10
converter_max = 1000


# Run system model for all inputs
training_df = mgridtest.multi_sim(solar_min, solar_max, battery_min, battery_max, converter_min, converter_max)
X = np.array([training_df['solar'], training_df['battery'], training_df['converter']])
X = X.transpose()
y_demand = np.array([training_df['Demand']]).T
y_cost = np.array([training_df['Cost']]).T
 
# Initialize model for predicting lifetime demand using NN
demand_model = regression_model(X, y_demand, lr) # Create regression model object
demand_model.init_reg_vars() # Regularize inputs and store regularization variables 
demand_model.torch_inputs() # Convert numpy inputs to torch tensors

# Initialize model for predicting lifetime cost using NN
cost_model = regression_model(X, y_cost, lr) # Create regression model object
cost_model.init_reg_vars() # Regularize inputs and store regularization variables 
cost_model.torch_inputs() # Convert numpy inputs to torch tensors

# Neural Network model for both demand and total cost 
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 3, 100, 1

# Construct our model by instantiating the class defined above
demand_nnmodel = TwoLayerNet(D_in, H, D_out)
cost_nnmodel = TwoLayerNet(D_in, H, D_out)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
demand_optimizer = torch.optim.SGD(demand_nnmodel.parameters(), lr=1e-3)
cost_optimizer = torch.optim.SGD(cost_nnmodel.parameters(), lr=1e-3)
for t in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    demand_pred = demand_nnmodel(demand_model.X_reg.type(torch.FloatTensor))
    cost_pred = cost_nnmodel(cost_model.X_reg.type(torch.FloatTensor))

    # Compute and print loss
    demand_loss = criterion(demand_pred, demand_model.y_reg.type(torch.FloatTensor))
    cost_loss = criterion(cost_pred, cost_model.y_reg.type(torch.FloatTensor))
    print(t, demand_loss.item(), cost_loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    demand_optimizer.zero_grad()
    cost_optimizer.zero_grad()
    demand_loss.backward()
    cost_loss.backward()
    demand_optimizer.step()
    cost_optimizer.step()
  
# Generate x(solar size) and y(storage size) values for plotting. Save deregularized values
x1=np.linspace(solar_min,solar_max,50)
x2=np.linspace(battery_min, battery_max,50)
x3=np.linspace(converter_min, converter_max)
X1, X2, X3 = np.meshgrid(x1, x2, x3)
X1_resh = np.reshape(X1, X1.shape[0]*X1.shape[1])
X2_resh = np.reshape(X2, X2.shape[0]*X2.shape[1])
X3_resh = np.reshape(X3, X3.shape[0]*X3.shape[1])
X_comb = np.array([X1_resh, X2_resh, X3_resh]).T

X_comb = torch.from_numpy(demand_model.regularize(X_comb, demand_model.reg_features['X_std'], demand_model.reg_features['X_mean'])).type(torch.FloatTensor)

# Create plot object and plot demand prediction
fig = plt.figure(figsize=(10.5,8))
ax = fig.add_subplot(111, projection='3d', xlabel = 'Solar Size', ylabel='Storage Size', zlabel='Demand (Wh)', aspect='equal')

Z = demand_nnmodel(X_comb)
Z = demand_model.dereg(Z, demand_model.reg_features['y_std'], demand_model.reg_features['y_mean'])
Z = torch.reshape(Z, (50,50)).detach().numpy()
ax.plot_wireframe(X, Y, Z, color='black')
plt.tight_layout()

# Create plot object and plot cost prediction
fig = plt.figure(figsize=(10.5,8))
ax = fig.add_subplot(111, projection='3d', xlabel = 'Solar Size', ylabel='Storage Size', zlabel='Cost($)', aspect='equal')

Z = cost_nnmodel(X_comb)
Z = cost_model.dereg(Z, cost_model.reg_features['y_std'], cost_model.reg_features['y_mean'])
Z = torch.reshape(Z, (50,50)).detach().numpy()
ax.plot_wireframe(X, Y, Z, color='black')
plt.tight_layout()



