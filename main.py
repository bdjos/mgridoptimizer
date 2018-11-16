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


############### Input System Info #####################
# Demand Specs
file = os.path.join('data', 'dc_foods_2014.csv')

# Define all input ranges. min == max == 0 for no component. min == max == value for single component
solar_min = 0
solar_max = 0
battery_min = 0
battery_max = 3000
converter_min = 0
converter_max = 3000

# Num steps for component sizing between min and max. 
# Recommended min numsteps =3, max numsteps = 10
# Higher = better accuracy, slower simulation time
# Lower = lower accuracy, faster simulation time
numsteps = 10

# Financials Specs
project_years = 10
interest = 0.06
inflation = 0.02

# Solar Specs
solar_base_cost = 10000
solar_power_cost = 1.60*1000

# Storage Specs
battery_soc_min = 0.2
battery_soc_max = 0.9
battery_efficiency = 0.95
battery_base_cost = 10000
battery_energy_cost = 0.5

# Converter Specs
converter_base_cost =  5000
converter_power_cost = 0.5

# Grid Specs
grid_cost = 0.35 / 1000  

#########################################################


# Run system model for all inputs
#training_df = mgridtest.multi_sim(file, solar_min, solar_max, battery_min, battery_max, converter_min, converter_max, numsteps,
#                                  project_years, interest, inflation, solar_base_cost, solar_power_cost, battery_soc_min, battery_soc_max, 
#                                  battery_efficiency, battery_base_cost, battery_energy_cost, converter_base_cost, 
#                                  converter_power_cost, grid_cost)

# Convert 
X = np.array([training_df['solar'], training_df['battery'], training_df['converter']])
X = X.transpose()
y_demand = np.array([training_df['Demand']]).T
y_cost = np.array([training_df['Cost']]).T

lr = 1e-4 # Set learning rate
 
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
criterion = torch.nn.MSELoss(reduction='sum') # Loss Function
demand_optimizer = torch.optim.SGD(demand_nnmodel.parameters(), lr=1e-4) 
cost_optimizer = torch.optim.SGD(cost_nnmodel.parameters(), lr=1e-4)

for t in range(1000):
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
x1=np.linspace(solar_min,solar_max,25)
x2=np.linspace(battery_min, battery_max,25)
x3=np.linspace(converter_min, converter_max,25)
X1, X2, X3 = np.meshgrid(x1, x2, x3)
X1_resh = np.reshape(X1, X1.shape[0]*X1.shape[1]*X1.shape[2])
X2_resh = np.reshape(X2, X2.shape[0]*X2.shape[1]*X2.shape[2])
X3_resh = np.reshape(X3, X3.shape[0]*X3.shape[1]*X3.shape[2])
X_comb = np.array([X1_resh, X2_resh, X3_resh]).T

X_comb_reg = torch.from_numpy(demand_model.regularize(X_comb, demand_model.reg_features['X_std'], demand_model.reg_features['X_mean'])).type(torch.FloatTensor)

# Create plot object and plot demand prediction
#fig = plt.figure(figsize=(10.5,8))
#ax = fig.add_subplot(111, projection='3d', xlabel = 'Solar Size', ylabel='Storage Size', zlabel='Demand (Wh)', aspect='equal')

Z_demand = demand_nnmodel(X_comb_reg)
Z_demand = demand_model.dereg(Z_demand, demand_model.reg_features['y_std'], demand_model.reg_features['y_mean'])
#Z = torch.reshape(Z, (50,50)).detach().numpy()
#ax.plot_wireframe(X, Y, Z, color='black')
#plt.tight_layout()

# Create plot object and plot cost prediction
#fig = plt.figure(figsize=(10.5,8))
#ax = fig.add_subplot(111, projection='3d', xlabel = 'Solar Size', ylabel='Storage Size', zlabel='Cost($)', aspect='equal')

Z_cost = cost_nnmodel(X_comb_reg)
Z_cost = cost_model.dereg(Z_cost, cost_model.reg_features['y_std'], cost_model.reg_features['y_mean'])
#Z = torch.reshape(Z, (50,50)).detach().numpy()
#ax.plot_wireframe(X, Y, Z, color='black')
#plt.tight_layout()

# Plot grid demand and cost for all input configurations
plt.figure(1, figsize=(9,3))
plt.subplot(211)
plt.plot(Z_demand.detach().numpy())
plt.subplot(212)
plt.plot(Z_cost.detach().numpy())
plt.show()

# Find minimum cost
index = np.argmin(Z_cost.detach().numpy())
min_cost = X_comb[index]

# Find lowest 10 values
#Create dataframe for inputs and results
input_df = pd.DataFrame(data=X_comb, columns=['Solar', 'Battery', 'Converter'])
output_df = pd.DataFrame(data=Z_cost.detach().numpy(), columns=['Cost'])

output_df = input_df.join(output_df)
output_df = output_df.sort_values(by=['Cost'])
output_df = output_df.reset_index(drop=True)