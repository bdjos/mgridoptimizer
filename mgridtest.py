# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 10:32:23 2018

@author: BJoseph
"""

from modules.mgrid_model import *
import os

# Solar Specs
solar_power = 100
solar_base_cost = 10000
solar_power_cost = 1.5*1000

# Storage Specs
battery_capacity = 100
battery_soc_min = 10
battery_soc_max = 90
battery_efficiency = 0.95
battery_base_cost = 10000
battery_energy_cost = 1*500

# Converter Specs
converter_power = 100
converter_base_cost =  1000
converter_power_cost = 100

# Create component objects
conv1 = converter(converter_power, converter_base_cost, converter_power_cost)
sol1 = solar.run_api(solar_power, solar_base_cost, solar_power_cost)
bat1 = battery(battery_capacity, battery_soc_min, battery_soc_max, 
               battery_efficiency, battery_base_cost, battery_energy_cost)
cont1 = controller()

# Configure controller battery and converter
cont1.config_storage(bat1, 'bat1', 'solar_support')
cont1.config_converter(conv1)

# Create system and add components
file = os.path.join('data', 'dc_foods_2014.csv')
system = system_model.import_fminute(file)

system.add_component(conv1, 'conv1')
system.add_component(sol1, 'sol1')
system.add_component(bat1, 'bat1')
system.add_component(cont1, 'cont1')

system.simulate()


