# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 13:05:11 2018

@author: BJoseph
"""

import sys
import os
import json
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from modules.demanddata import import_data

class battery():
    def __init__(self, energy_capacity, soc_min, soc_max, efficiency, base_cost, energy_cost):
        self.type = 'battery'
        self.energy_capacity = energy_capacity * 1000
        self.energy_max = energy_capacity * 1000
        self.energy_rem = energy_capacity * 1000
        self.base_cost = base_cost
        self.energy_cost = energy_cost 
        
    def charge(self, amt):
        self.energy_rem += amt
    
    def cost_calc(self):
        return self.base_cost + self.energy_capacity*self.energy_cost      

class converter():
    def __init__(self, power, base_cost, power_cost):
        self.type = 'converter'
        self.power = power*1000
        self.base_cost = base_cost
        self.power_cost = power_cost
        self.capacity_rem = self.power
    
    def capacity_calc(self, amt):
        self.capacity_rem = self.capacity_rem - abs(amt)
    
    def reset_capacity(self):
        self.capacity_rem = self.power
        
    def cost_calc(self):
        return self.base_cost + self.power*self.power_cost   

class controller():
    def __init__(self):
        self.type = 'controller'
        self.converter = None
        self.battery_list = {}
        
    def config_converter(self, converter):
        self.converter = converter
        
    def config_storage(self, battery, name, mode):
#        if battery.idx in battery_list:
#            storage[battery.idx]
        if mode == 'solar_support':
            configs = None
        if mode == 'arbitrage':
            configs = {'buy_range': buy_range, 'sell_range': sell_range}
        
        self.battery_list[name] = {'object': battery, 'mode': mode, 'configs': configs}
    
    def check_solar_support(self, battery, amt):
        if amt > 0: # Charge if amt > 0
            if amt < battery.energy_capacity - battery.energy_rem:
                return amt
            else:
                return battery.energy_capacity - battery.energy_rem
        elif amt < 0: # Discharge if amt < 0
            if abs(amt) < battery.energy_rem - 0:
                return amt
            else:
                return battery.energy_rem - 0
        else:
            return 0
    
    def check_converter(self, amt):
        if abs(amt) < self.converter.capacity_rem:
            return amt
        else:
            return abs(amt)/amt * self.converter.capacity_rem
                    
    def io(self, amt):
        self.converter.reset_capacity()
        for battery in self.battery_list:
            if self.battery_list[battery]['mode'] == 'solar_support':
                charge = self.check_solar_support(self.battery_list[battery]['object'], amt)
            elif self.battery_list[battery]['mode'] == 'arbitrage':
                charge = self.check_arbitrage(battery['object'], amt)
            elif self.battery_list[battery]['mode'] == 'peak_shaving':
                charge = self.check_peak_shaving(battery['object'], amt)
            charge = self.check_converter(charge)
            self.converter.capacity_calc(charge) # Update Converter capacity available
            self.battery_list[battery]['object'].charge(charge) # Updte battery capacity
            amt = amt - charge    
        return amt

    def __str__(self):
        return f"{self.__dict__}"
    
class solar:
    def __init__(self, solar_df, system_capacity, base_cost, perw_cost):
        self.type = 'solar'
        self.solar_df = solar_df
        self.system_capacity = system_capacity
        self.base_cost = base_cost
        self.perw_cost = perw_cost
    
    @classmethod
    def run_api(cls, system_capacity, base_cost, perw_cost, output_format = 'json', api_key = 'NueBIZfkmWJUjd7MK5LRrIi7MhFL2LLuRlZswFcM',
                module_type = '0', losses = '10', array_type = '1', tilt = '30', azimuth = '180',
                lat = '43.6532', lon = '79.3832', radius = '0', timeframe = 'hourly', dc_ac_ratio = '1.1'):
    
        url = f'https://developer.nrel.gov/api/pvwatts/v5.{output_format}?api_key={api_key}&system_capacity={system_capacity}&module_type={module_type}&losses={losses}&array_type={array_type}&tilt={tilt}&azimuth={azimuth}&lat={lat}&lon={lon}&radius={radius}&timeframe={timeframe}&dc_ac_ratio={dc_ac_ratio}'
        with urllib.request.urlopen(url) as url:
            data = json.loads(url.read().decode())
            
        df = pd.DataFrame(data=data['outputs']['ac'], columns=['Production'])
        df.index.names = ['Hour']  
        return cls(df, system_capacity, base_cost, perw_cost)    

    def cost_calc(self):
        return self.base_cost + self.perw_cost*self.system_capacity*1000
                                                    
class system_model:  
    def __init__(self, demand_df):
        self.demand_obj = demand_df
        self.system_components = {} # Holds all system components
        
    @classmethod
    def import_fminute(cls, file):
        '''
        Import csv file and convert data to hourly demand data if 15 minute 
        intervals. Convert to pd dataframe
        '''
        demand_df = import_data.fifteenMinute(file)
        return cls(demand_df)
    
    def add_component(self, component, name):
        if name in self.system_components:
            print(f'{name} already in system components. Either rename or remove existing component')
        else:
            self.system_components[name] = component
                        
    def remove_component(self, name):
        if name in self.system_components:
            del self.system_components[name]
        else:
            print(f'{name} is not in system')
        
    def simulate(self):
        # Run passive production
        delta = pd.DataFrame(data=self.demand_obj.df['Demand'] - self.system_components['solar'].solar_df['Production'], columns=['Demand'])
        
        #Run controller
        storage_vals = []
        for demand_vals in delta['Demand']:
            storage_vals.append(-self.system_components['controller'].io(-demand_vals)) # Neg  values for demand
        
        self.simulated_df = pd.DataFrame(data=storage_vals, columns=['Demand'])
        self.simulated_df.index.names = ['Hour']
        
        return sum(self.simulated_df['Demand'])
                
    def plot_demands(capacity_map, demands):
        plt.plot(capacity_map, demands)
        plt.show()
        
    def __str__(self):
        
        components = ''
        for component in self.system_components:
            components = component + components
        return components

def regtrain_data(demand_file, solar_min, solar_max, solar_base_cost, solar_perw_cost, storage_min, 
                  storage_max, storage_base_cost, storage_energy_cost, converter_max, converter_base_cost, 
                  converter_power_cost, numsteps_solar, numsteps_storage, energy_cost, lifecycle):
    """
    Run system model on solar_min to solar_max & storage_min to storage_max. Simulate
    data on demand_file. Return results in pd df.
    """
    system_arch = system_model.import_fminute(demand_file) # Initialize system model archetype
    reg_vals = {'solar_capacity': [], 'storage_capacity': [], 'converter_capacity': [], 
                'system_cost': [], 'demand': [], 'lifecycle_cost': []}
    
    #Loop through all solar and storage capacities to create training data
    for solar_capacities in np.linspace(solar_min, solar_max, numsteps_solar):
        model_temp = copy.copy(system_arch)
        model_temp.add_component(solar_capacities, solar_base_cost, solar_perw_cost)
        for storage_capacities in np.linspace(storage_min, storage_max, numsteps_storage):
            model_temp.add_component(storage_capacities, storage_capacities, storage_base_cost, 
                  storage_power_cost, storage_energy_cost)
            for converter_capacities in np.linspace(converter_min, converter_max, numsteps_converter):
                model_temp.add_component()
                demand = model_temp.simulate()
            
                reg_vals['solar_capacity'].append(solar_capacities)
                reg_vals['storage_capacity'].append(storage_capacities)
                reg_vals['converter_capacity'].append(converter_capacities)
                reg_vals['system_cost'].append(model_temp.system_cost)
                reg_vals['demand'].append(demand)
                reg_vals['lifecycle_cost'].append(model_temp.system_cost + energy_cost*demand*lifecycle)
            
    training_df = pd.DataFrame(reg_vals)
    training_df.index.names = ['hours']
    return training_df   
     
#if __name__ == '__main__':
#    print('okay!')
#def regressionvals(demand_file, min_pv_size, max_pv_size, num_steps=3):
#    
#    # Merge DFs and generate consumption - production data sets to determine yearly
#    # energy consumption after PV install
#    df = pd.merge(demand_df, pv_df, on='Hour')
#    
#    demands = []
#    for i in range(len(pv_df.columns)):
#        df[f'Delta{i}'] = df['Demand'] - df[f'Production{i}']
#        df[f'Delta{i}'] = df[f'Delta{i}'].apply(lambda x: 0 if x<0 else x)
#        demand_tot = df[f'Delta{i}'].sum()
#        demands.append(demand_tot)
#    
#    return np.array(capacity_map), np.array(demands)
#        # Get PV data between min and max inputs
#    pv_df, capacity_map = run_api(min_pv_size, max_pv_size, num_steps)
#    # Get demand dataframe
#    demand_df = import_data.fifteenMinute(demand_file).df
#    
#    capacity_map = [1]
#    count = 1
#    
#    # Call pvwatts api on min system size = 1 to initialize
#    system_capacity=1
#    pv_df = get_pv(output_format, api_key, system_capacity, module_type, losses, array_type, tilt, azimuth, lat, lon, radius, timeframe, dc_ac_ratio)
#    pv_df.columns = ['Production0']
#    
#    # Call pvwatts api for num_steps between min and max size
#    for system_capacity in np.linspace(min_size, max_size, num_steps):
#        print(system_capacity)
#        temp_df = get_pv(output_format, api_key, str(system_capacity), module_type, losses, array_type, tilt, azimuth, lat, lon, radius, timeframe, dc_ac_ratio)
#        temp_df.columns = [f'Production{count}']
#        pv_df = pd.merge(pv_df, temp_df, on='Hour')
#        capacity_map.append(system_capacity)
#        count += 1
#    return pv_df, capacity_map