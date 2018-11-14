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


def run_model():
    
    energy_cost = 0.30
    
    # Solar Specs
    solar_range = np.linspace(10, 500, 5)
    solar_base_cost = 10000
    solar_power_cost = 1.5*1000
    
    # Storage Specs
    storage_range = np.linspace(10, 500, 5)
    battery_capacity = 500
    battery_soc_min = 10
    battery_soc_max = 90
    battery_efficiency = 0.95
    battery_base_cost = 10000
    battery_energy_cost = 1*500
    
    # Converter Specs
    converter_range = np.linspace(10, 500, 5)
    converter_power = 1000
    converter_base_cost =  1000
    converter_power_cost = 100
    
    # Create system and add components
    file = os.path.join('data', 'dc_foods_2014.csv')
    system = system_model.import_fminute(file)
    
    solar_objs = []
    battery_objs= []
    converter_objs = []
    
    # Create component objects for ranges:
    for i in solar_range:
        solar_objs.append(solar.run_api(i, solar_base_cost, solar_power_cost))
    
    for i in storage_range:
        battery_objs.append(battery(i, battery_soc_min, battery_soc_max,
                   battery_efficiency, battery_base_cost, battery_energy_cost))
    
    for i in converter_range:
        converter_objs.append(converter(i, converter_base_cost, converter_power_cost))
    
    output = []
    system_temp = copy.copy(system)
    # Combine all combinations of components 
    for combinations in product(solar_objs, battery_objs, converter_objs): 
        control = controller()
        # Add all components to system
        for component in combinations: 
            system_temp.add_component(component, component.type)
        system_temp.add_component(control, control.type)
        system_temp.system_components['controller'].config_storage(system_temp.system_components['battery'], system_temp.system_components['battery'].type, 'solar_support') ### HOW TO AUTO CONFIG BATTERIES
        system_temp.system_components['controller'].config_converter(system_temp.system_components['converter']) ### HOW TO AUTO CONFIG CONVERTER
        output.append(system_temp.simulate())
        # Clear system:
        for component in system_temp.system_components.copy():
            system_temp.remove_component(component)
        
    # Convert all component sizes to lists
    component_sizes = {'solar': [], 'battery': [], 'converter': []}
    for size in product(solar_range, storage_range, converter_range):
        component_sizes['solar'].append(size[0])
        component_sizes['battery'].append(size[1])
        component_sizes['converter'].append(size[2])
    
    df = pd.DataFrame(data=component_sizes)
    df['demand'] = output
    return df

lr = 1e-3
# Get system sizes and demands from PV Watts API. Returns numpy array

training_df = mgridtest.multi_sim()
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



