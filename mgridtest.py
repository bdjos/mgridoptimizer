# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 10:32:23 2018

@author: BJoseph
"""

from modules.mgrid_model import *
from modules.demanddata import import_data
import os
import numpy as np
import pandas as pd
from itertools import product
import copy


def multi_simulate():
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
    df['output'] = output

# Demand Specs
file = os.path.join('data', 'dc_foods_2014.csv')

# Solar Specs
solar_capacity = 500
solar_base_cost = 10000
solar_power_cost = 1.5*1000

# Storage Specs
battery_capacity = 500
battery_soc_min = 10
battery_soc_max = 90
battery_efficiency = 0.95
battery_base_cost = 10000
battery_energy_cost = 0.5

# Converter Specs
converter_power = 1000
converter_base_cost =  1000
converter_power_cost = 1

# Grid Specs
grid_cost = 0.30 / 1000  
    
# Create system model object
system = system_model()    

# Create components
demand = import_data.fifteenMinute(file)
bat1 = battery(battery_capacity, battery_soc_min, battery_soc_max, battery_efficiency, battery_base_cost, battery_energy_cost)
sol1 = solar.run_api(solar_capacity, solar_base_cost, solar_power_cost)
conv1 = converter(converter_power, converter_base_cost, converter_power_cost)
grid1 = grid(grid_cost)
cont1 = controller()

# Add components to system
system.add_component(demand, 'demand', 'stage0')
system.add_component(sol1, 'sol1', 'stage0')
system.add_component(bat1, 'bat1', 'stage1')
system.add_component(conv1, 'conv1', 'stage1')
system.add_component(grid1, 'grid1', 'stage2')
system.add_component(cont1, 'cont1')

# Configure controller
cont1.config_converter(conv1)
cont1.config_storage(bat1, 'bat1', mode='solar_support')

# Simulate system
system.simulate()
print(system.total_costs())

