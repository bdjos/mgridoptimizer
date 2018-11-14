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


def multi_sim():
    # Project Specs
    project_years = 20
    
    # Demand Specs
    file = os.path.join('data', 'dc_foods_2014.csv')
    
    # Solar Specs
    solar_base_cost = 10000
    solar_power_cost = 1.5*1000
    
    # Storage Specs
    battery_soc_min = 10
    battery_soc_max = 90
    battery_efficiency = 0.95
    battery_base_cost = 10000
    battery_energy_cost = 0.5
    
    # Converter Specs
    converter_base_cost =  1000
    converter_power_cost = 1
    
    # Grid Specs
    grid_cost = 0.30 / 1000  
        
    # Equipment Ranges
    solar_range = np.linspace(10, 500, 3)
    storage_range = np.linspace(10, 500, 3)
    converter_range = np.linspace(10, 500, 3)
    
    # Create system model object
    system = system_model()    

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
    
    output = {
            'Demand': [],
            'Cost': []
            }
    
    system_temp = copy.copy(system)
    demand = import_data.fifteenMinute(file)
    
    # Combine all combinations of components 
    for combinations in product(solar_objs, battery_objs, converter_objs): 
        # Define and add demand, controller and grid to system
        cont1 = controller()
        grid1 = grid(grid_cost, project_years)
        system_temp.add_component(demand, 'demand', 'stage0')
        system_temp.add_component(cont1, 'cont1')
        system_temp.add_component(grid1, 'grid1', 'stage2')
        
        # Add all variable components to system
        for component in combinations: 
            if component.type == 'solar':
                stage = 'stage0'
            elif component.type == 'battery':
                stage = 'stage1'
            elif component.type == 'converter':
                stage = 'stage1'
            system_temp.add_component(component, component.type, stage)
        
        print(system_temp.system_hierarchy)
        # Configure controller
        cont1.config_storage(system_temp.system_components['battery'], system_temp.system_components['battery'].type, 'solar_support') ### HOW TO AUTO CONFIG BATTERIES
        cont1.config_converter(system_temp.system_components['converter']) ### HOW TO AUTO CONFIG CONVERTER
        system_temp.simulate()
        output['Demand'].append(sum(grid1.total_supply))
        output['Cost'].append(system_temp.total_costs())
        
        # Clear system:
        system_temp.clear()
        
    # Convert all component sizes to lists
    component_sizes = product(solar_range, storage_range, converter_range)
    df = pd.DataFrame(data=list(component_sizes), columns=('solar', 'battery', 'converter'))
    df = pd.concat([df, pd.DataFrame(output)], axis=1)
    return df

def single_sim(converter_power):
    # Project Specs
    project_years = 20
    
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
    converter_power = converter_power
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
    grid1 = grid(grid_cost, project_years)
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
    return system



