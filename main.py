# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 15:29:25 2018

@author: BJoseph
"""
from regressionvals import regressionvals
from regressionmodel import regression_model
import numpy as np
import matplotlib.pyplot as plt
import torch

# Inputs for PV Watts API
system_cost = 2.5*1000 # $/kW cost of system
energy_cost = 0.25/1000 # $/kWh cost of electricty
demand_file = 'dc_foods_2014.csv'
min_pv_size = 100
max_pv_size = 2000
num_steps = 5 # min 3

# Get system sizes and demands from PV Watts API. Returns numpy array
system_sizes, demands = regressionvals(demand_file, min_pv_size, max_pv_size, num_steps)

# Regularize data for inputting into regression model
#demands_reg, demands_std, demands_mean = regression_model.regularize(demands)

# Initialize regression model using numpy to pytorch method torch_inputs
reg_model = regression_model(system_sizes, demands, lr=1e-1)

# Initialize theta values with random numbers and regularize inputs/outputs 
reg_model.init_theta()
reg_model.init_reg_vars()
reg_model.torch_inputs()

# Run gradient descent
a, b, c, d = reg_model.grad_descent()

# Evaluate and deregularize initial test values
y_init = regression_model.lin(a, b, c, d, reg_model.X_reg)
y_hat = np.array(regression_model.dereg(y_init, reg_model.y_std, reg_model.y_mean))

# Plot predicted and calculated values of system size vs electricity demand
plt.plot(system_sizes, y_hat)
plt.plot(system_sizes, demands)
plt.show()

# Define cost function for 20 year cost of system + electricity
x = np.arange(min_pv_size, max_pv_size, 10)
y_init = regression_model.lin(a.item(), b.item(), c.item(), d.item(), np.array(regression_model.regularize(x, reg_model.X_std, reg_model.X_mean)))
y_hat = np.array(reg_model.dereg(y_init, reg_model.y_std, reg_model.y_mean))
cost = system_cost*x + energy_cost*(y_hat)*20

plt.plot(x, cost)
plt.show()

# Gradient descent on the cost function
def cost_func_trunc(a, b, c, d, X_std, X_mean, y_std, y_mean, theta):
    cost = system_cost*theta + energy_cost*regression_model.dereg(regression_model.lin(a, b, c, d, regression_model.regularize(theta, reg_model.X_std, reg_model.X_mean)), y_std, y_mean)*20
    return cost


# Initialize theta to median of initial X set    
theta = torch.tensor([reg_model.X.median()])
theta.requires_grad_()
#loss = cost_func_trunc(a.item(), b.item(), c.item(), d.item(), reg_model.X_std, reg_model.X_mean, reg_model.y_std, reg_model.y_mean, theta)
theta = reg_model.standard_grad_descent(cost_func_trunc, theta, lr=0.1)
print(theta)
