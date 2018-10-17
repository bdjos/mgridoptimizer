# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 13:05:11 2018

@author: BJoseph
"""

import json
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from demanddata import import_data
from pvwattsapi import get_pv

class storage:
    def __init__(self, power, energy_capacity):
        self.power = power * 1000
        self.energy_max = energy_capacity * 1000
        self.energy_rem = 0
    
    def discharge(self, amt):
        if amt > 0:
            # If facility demand is positive, discharge battery (discharge=positive)
            discharged = min(self.energy_rem, self.power, amt)
            self.energy_rem -= discharged
        elif amt < 0:
            # if facility demand is negative (solar production greater than demand), charge battery
            # (discharge=negative)
            discharged = -1*min((self.energy_max-self.energy_rem), self.power, abs(amt))
            self.energy_rem -= discharged 
        else:
            discharged = 0
        return discharged
        
class solar:
    def __init__(self, solar_df, system_capacity):
        self.solar_df = solar_df
        self.system_capacity = system_capacity
    
    @classmethod
    def run_api(cls, system_capacity, output_format = 'json', api_key = 'NueBIZfkmWJUjd7MK5LRrIi7MhFL2LLuRlZswFcM',
                module_type = '0', losses = '10', array_type = '1', tilt = '30', azimuth = '180',
                lat = '43.6532', lon = '79.3832', radius = '0', timeframe = 'hourly', dc_ac_ratio = '1.1'):
    
        url = f'https://developer.nrel.gov/api/pvwatts/v5.{output_format}?api_key={api_key}&system_capacity={system_capacity}&module_type={module_type}&losses={losses}&array_type={array_type}&tilt={tilt}&azimuth={azimuth}&lat={lat}&lon={lon}&radius={radius}&timeframe={timeframe}&dc_ac_ratio={dc_ac_ratio}'
        with urllib.request.urlopen(url) as url:
            data = json.loads(url.read().decode())
            
        df = pd.DataFrame(data=data['outputs']['ac'], columns=['Production'])
        df.index.names = ['Hour']  
        return cls(df, system_capacity)                                    
    
class system_model:  
    def __init__(self, demand_df):
        self.demand_obj = demand_df
        self.storage_obj = storage(0, 0)
        self.solar_obj = None
    
    @classmethod
    def import_fminute(cls, file):
        '''
        Import csv file and convert data to hourly demand data if 15 minute 
        intervals. Convert to pd dataframe
        '''
        demand_df = import_data.fifteenMinute(file)
        return cls(demand_df)
    
    def add_solar(self, system_capacity, overwrite=False):
        if self.solar_obj and overwrite==False:
            print('''Solar already added to system. To modify, use add_solar method
                  with overwrite==True''')
            return None
        else:
            self.solar_obj = solar.run_api(system_capacity)
            print(f'{system_capacity} kW solar added to system')
            
    def remove_solar(self):
        if not self.solar_obj:
            print('No solar added to system')
            return None
        else:
            self.solar_obj = None
    
    def add_storage(self, power, energy_capacity, overwrite=False):
        if self.storage_obj and overwrite==False:
            self.storage_obj = storage(power, energy_capacity)
#            print('''Storage already added to system. To modify, use remove_storage method 
#                  and add new storage''')
        else:
            self.storage_obj = storage(power, energy_capacity)
            print(f'{power} W, {energy_capacity} Wh storage added to system')
        
    def remove_storage(self):
        if not self.storage_obj:
            print('No storage added to system')
            return None
        else:
            self.storage_obj = None
        
    def simulate(self):
        delta = pd.DataFrame(data=self.demand_obj.df['Demand'] - self.solar_obj.solar_df['Production'], columns=['Demand'])
        
        storage_vals = []
        for demand_vals in delta['Demand']:
            storage_vals.append(self.storage_obj.discharge(demand_vals))
        
        self.simulated_df = pd.DataFrame(data=storage_vals, columns=['Demand'])
        self.simulated_df.index.names = ['Hour']
        self.simulated_df['Demand'] = delta['Demand'] - self.simulated_df['Demand']
        
        self.simulated_df['Demand'] = self.simulated_df['Demand'].apply(lambda x: 0 if x < 0 else x)
        return sum(self.simulated_df['Demand'])
            
        
    def plot_demands(capacity_map, demands):
        plt.plot(capacity_map, demands)
        plt.show()
        
    def __str__(self):
        
        return(f'''
        Solar system size: {self.solar_obj.system_capacity} kW
        Storage power: {self.storage_obj.power} W
        Storage capacity: {self.storage_obj.energy_max} Wh
        ''')

def regression_training(demand_file, solar_min, solar_max, storage_min, storage_max, numsteps):
    """
    Run system model on solar_min to solar_max & storage_min to storage_max. Simulate
    data on demand_file. Return results in pd df.
    """
    system_arch = system_model.import_fminute(demand_file) # Initialize system model archetype
    reg_vals = {'solar_capacity': [], 'storage_capacity': [], 'demand': []}
    
    for solar_capacities in np.linspace(solar_min, solar_max, numsteps):
        model_temp = copy.copy(system_arch)
        model_temp.add_solar(solar_capacities)
        for storage_capacities in np.linspace(storage_min, storage_max, numsteps):
            model_temp.add_storage(storage_capacities, storage_capacities)
            demand = model_temp.simulate()
            
            reg_vals['solar_capacity'].append(solar_capacities)
            reg_vals['storage_capacity'].append(storage_capacities)
            reg_vals['demand'].append(demand)
            
    training_df = pd.DataFrame(reg_vals)
    training_df.index.names = ['hours']
    return training_df   
     
if __name__ == '__main__':
    
    def single_system_test(solar_power, storage_power, storage_capacity):
        system = system_model.import_fminute('dc_foods_2014.csv')
        system.add_solar(solar_power)
        system.add_storage(storage_power, storage_capacity)
        system.view_system()
        total = system.simulate()
    
    training_df = training_vals('dc_foods_2014.csv', 300, 500, 100, 300, 3)
                
                
        
    
def regressionvals(demand_file, min_pv_size, max_pv_size, num_steps=3):
    
    # Merge DFs and generate consumption - production data sets to determine yearly
    # energy consumption after PV install
    df = pd.merge(demand_df, pv_df, on='Hour')
    
    demands = []
    for i in range(len(pv_df.columns)):
        df[f'Delta{i}'] = df['Demand'] - df[f'Production{i}']
        df[f'Delta{i}'] = df[f'Delta{i}'].apply(lambda x: 0 if x<0 else x)
        demand_tot = df[f'Delta{i}'].sum()
        demands.append(demand_tot)
    
    return np.array(capacity_map), np.array(demands)
        # Get PV data between min and max inputs
    pv_df, capacity_map = run_api(min_pv_size, max_pv_size, num_steps)
    # Get demand dataframe
    demand_df = import_data.fifteenMinute(demand_file).df
    
    capacity_map = [1]
    count = 1
    
    # Call pvwatts api on min system size = 1 to initialize
    system_capacity=1
    pv_df = get_pv(output_format, api_key, system_capacity, module_type, losses, array_type, tilt, azimuth, lat, lon, radius, timeframe, dc_ac_ratio)
    pv_df.columns = ['Production0']
    
    # Call pvwatts api for num_steps between min and max size
    for system_capacity in np.linspace(min_size, max_size, num_steps):
        print(system_capacity)
        temp_df = get_pv(output_format, api_key, str(system_capacity), module_type, losses, array_type, tilt, azimuth, lat, lon, radius, timeframe, dc_ac_ratio)
        temp_df.columns = [f'Production{count}']
        pv_df = pd.merge(pv_df, temp_df, on='Hour')
        capacity_map.append(system_capacity)
        count += 1
    return pv_df, capacity_map