# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 15:29:25 2018

@author: BJoseph
"""

import sys
sys.path.append('/models')

import os
from modules.mgrid_model import *
from modules.regression_model import *
from modules.nn_model import *
import numpy as np
import matplotlib.pyplot as plt
import torch

# Inputs for PV Watts API
solar_base_cost = 10000
solar_perw_cost = 1.3 # $/W cost of system
storage_base_cost = 10000
storage_power_cost = 0.4
storage_energy_cost = 0.4
energy_cost = 0.30/1000 # $/kWh cost of electricity
file = os.path.join('data', 'dc_foods_2014.csv')
min_pv_size = 10
max_pv_size = 5000
min_storage_size = 10
max_storage_size = 5000
numsteps_solar = 10 # min 3
numsteps_storage=20
lifecycle=20

# Get system sizes and demands from PV Watts API. Returns numpy array

#training_df = regtrain_data(file, min_pv_size, max_pv_size, solar_base_cost, solar_perw_cost, 
#                            min_storage_size, max_storage_size, storage_base_cost, storage_power_cost, 
#                            storage_energy_cost, numsteps_solar, numsteps_storage, energy_cost, lifecycle)
X = np.array([training_df['solar_capacity'], training_df['storage_capacity']])
X = X.transpose()
y_demand = np.array([training_df['demand']]).T
y_cost = np.array([training_df['lifecycle_cost']]).T

def hypothesis(theta, X):
   "Hypothesis function for overall energy demand"
   X = regression_model.feature_expand(X)
   return torch.mm(X, theta.transpose(0,1))[:,0]
 
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
N, D_in, H, D_out = 64, 2, 100, 1

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
x=np.linspace(min_pv_size,max(max_pv_size, max_storage_size),50)
y=np.linspace(min_storage_size,max(max_pv_size,max_storage_size),50)
X, Y = np.meshgrid(x, y)
X_resh = np.reshape(X, X.shape[0]*X.shape[1])
Y_resh = np.reshape(Y, Y.shape[0]*Y.shape[1])
X_comb = np.array([X_resh, Y_resh]).T

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



