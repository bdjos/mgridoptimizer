# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 13:05:11 2018

@author: BJoseph
"""

import sys
import os
sys.path.insert(0, os.path.join('..'))
import json
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
from mgridoptimizer.modules.demanddata import import_data

class Cost_Component():
    # Used to define a cost calculation from a component. Build this out as 
    # parent obj to all cost components to simplify code
    def __init__(self, **costs):
        self.cost_list = costs
    
    def cost_calc(self, years):
        cost = self.build_costs + self.yearly_costs * years

class Demand:
    def __init__(self, df):
        '''
        Return pandas dataframe object for a facility's electricity demand
        '''
        self.demand = df
        self.cost_component = False

    @classmethod
    def fifteenMinute(cls, file):
        '''
        Import csv file and convert data to hourly demand data if 15 minute
        intervals. Convert to pd dataframe
        '''

        # Import from CSV
        df = pd.read_csv(file, names=['Demand'])

        # Convert to Watts
        df['Demand'] = df['Demand'] * 1000

        # Create list of hours for 15 minute interval data to convert to hourly data
        hour_list = []
        for hour in range(8760):
            for count in range(4):
                hour_list.append(hour)

        df['Hour'] = hour_list

        # Convert to hourly data: Group by hours and average to find hourly energy consumption
        df = df.groupby(by='Hour').mean()

        # Check length of dataframe
        print(cls.checkLength(df))
        return cls(list(-df['Demand']))

    @classmethod
    def hourlyInterval(cls, file):
        '''
        Import csv file and convert to pd dataframe
        '''
        df = pd.read_csv(file, names=['Demand'])
        df.index.names = ['Hour']

        # Check length of dataframe
        print(cls.checkLength(df))

        return cls(df)

    @classmethod
    def checkLength(cls, df):
        '''
        Check length of dataframe to ensure that length is one year (8760 hours).
        Return warning message if length is above/below this value
        '''

        if len(df) < 8760:
            warning = f'''Dataframe length is less than 8760 hours ({len(df)} 
            points). If you continue, analysis will occur on only the first
            {len(df)} points, corresponding to Jan-1 00:00:00 onwards. This may 
            give a sub-optimal result.
            '''
        elif len(df) > 8760:
            warning = f'''Dataframe length is greater than 8760 hours ({len(df)} 
            points). If you continue, analysis will occur on the first 8760 points, 
            corresponding to Jan-1 00:00:00 to Dec-31 23:00:00))This may give a 
            sub-optimal result.
            '''
        else:
            warning = f'''8760 values found'''

        return warning

    def cost_calc(self):
        return 0

    def output(self):
        return {'demand': self.demand}

class Battery():
    def __init__(self, energy_capacity, soc_min, soc_max, efficiency, base_cost, energy_cost):
        self.type = 'battery'
        self.energy_capacity = energy_capacity * soc_max * 1000
        self.floor = energy_capacity * soc_min * 1000
        self.energy_max = energy_capacity * 1000
        self.energy_rem = energy_capacity * 1000
        self.base_cost = base_cost
        self.cost_component = True
        self.energy_cost = energy_cost
        self.counter = 0 # Counter for charge
        self.stats = {
            'hour': [],
            'output': [],
            'levels': [],
            'soc': []
            }
        self.cost_list = {
                'base_cost': self.base_cost + self.energy_capacity*self.energy_cost,
                'yearly_cost': 0
                }
        
    def charge(self, amt):
        self.energy_rem += amt
        self.stats['hour'].append(self.counter)
        self.stats['output'].append(amt)
        self.stats['levels'].append(self.energy_rem)
        self.stats['soc'].append(self.energy_rem/self.energy_max)
        self.counter += 1
        
    def output(self):
        return self.stats

class Converter():
    def __init__(self, power, base_cost, power_cost):
        self.type = 'converter'
        self.power = power*1000
        self.base_cost = base_cost
        self.power_cost = power_cost
        self.cost_component = True
        self.capacity_rem = self.power
        self.stats = {
                'output': []
                }
        self.cost_list = {
                'base_cost': self.base_cost + self.power*self.power_cost,
                'yearly_cost': 0
                }
    
    def capacity_calc(self, amt):
        self.capacity_rem = self.capacity_rem - abs(amt)
        self.stats['output'].append(amt)
    
    def reset_capacity(self):
        self.capacity_rem = self.power

    def output(self):
        return self.stats

class Controller():
    """
    Controls input and output of system stage 2 (controllable input/output such as generator/storage)
    """
    def __init__(self, active):
        self.active = active
        self.type = 'controller'
        self.converter = None
        self.battery_list = {}
        self.cost_component = False
        self.cost_list = {
                'base_cost': 0,
                'yearly_cost': 0
                }
        
    def config_component(self, component_object, name, mode):
        if component_object.type == 'battery':
            if mode == 'ss':
                configs = None
            if mode == 'ab':
                configs = {'buy_range': buy_range, 'sell_range': sell_range}

            self.battery_list[name] = {'object': component_object, 'mode': mode, 'configs': configs}

        elif component_object.type == 'converter':
            self.converter = component_object

    def check_solar_support(self, battery, amt):
        if amt > 0: # Charge if amt > 0
            if amt < battery.energy_capacity - battery.energy_rem:
                return amt
            else:
                return battery.energy_capacity - battery.energy_rem
        elif amt < 0: # Discharge if amt < 0
            if abs(amt) < battery.energy_rem - battery.floor:
                return amt
            else:
                return battery.energy_rem - battery.floor
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
            if self.battery_list[battery]['mode'] == 'ss':
                charge = self.check_solar_support(self.battery_list[battery]['object'], amt)
            elif self.battery_list[battery]['mode'] == 'ab':
                charge = self.check_arbitrage(battery['object'], amt)
            elif self.battery_list[battery]['mode'] == 'ps':
                charge = self.check_peak_shaving(battery['object'], amt)
            charge = self.check_converter(charge)
            self.converter.capacity_calc(charge) # Update Converter capacity available
            self.battery_list[battery]['object'].charge(charge) # Update battery capacity
            amt = amt - charge    
        return amt
    
    def cost_calc(self):
        return 0

    def output(self):
        return None
    
    # def __str__(self):
    #     return f"{self.__dict__}"
    
class Solar():
    def __init__(self, solar_df, json_demand, system_capacity, base_cost, perw_cost):
        self.type = 'solar'
        self.demand = solar_df
        self.json_demand = json_demand
        self.system_capacity = system_capacity
        self.base_cost = base_cost
        self.perw_cost = perw_cost
        self.cost_component = True
        self.cost_list = {
                'base_cost': self.base_cost + self.perw_cost*self.system_capacity,
                'yearly_cost': 0
                }
    
    @classmethod
    def run_api(cls, system_capacity, base_cost, perw_cost, output_format = 'json', api_key = 'NueBIZfkmWJUjd7MK5LRrIi7MhFL2LLuRlZswFcM',
                module_type = '0', losses = '10', array_type = '1', tilt = '30', azimuth = '180',
                lat = '43.6532', lon = '79.3832', radius = '0', timeframe = 'hourly', dc_ac_ratio = '1.1'):
    
        url = f'https://developer.nrel.gov/api/pvwatts/v5.{output_format}?api_key={api_key}&system_capacity={system_capacity}&module_type={module_type}&losses={losses}&array_type={array_type}&tilt={tilt}&azimuth={azimuth}&lat={lat}&lon={lon}&radius={radius}&timeframe={timeframe}&dc_ac_ratio={dc_ac_ratio}'
        with urllib.request.urlopen(url) as url:
            data = json.loads(url.read().decode())
            
        df = pd.DataFrame(data=data['outputs']['ac'], columns=['Production'])
        df.index.names = ['Hour']  
        return cls(list(df['Production']), data['outputs']['ac'], system_capacity, base_cost, perw_cost)

    def output(self):
        return {'production': self.demand}

class Grid():
    "Grid component for modelling grid input to system"
    def __init__(self, energy_cost, nm = False):
        self.type = 'grid'
        self.nm = nm
        self.energy_cost = energy_cost
        self.total_supply = []
        self.cost_component = True
        self.cost_list = {
                'base_cost': 0,
                'yearly_cost': 0
                }
        
    def supply(self, amt):
        self.total_supply = [-x for x in amt]
    
    def cost_calc(self):
        if self.nm == True: 
            self.cost_list['yearly_cost'] = self.energy_cost * sum(self.total_supply)
        else: # price only supplied energy (positive values) if net metering not allowed
            self.cost_list['yearly_cost'] = self.energy_cost * sum([x for x in self.total_supply if x > 0])

    def output(self):
        return self.total_supply
                                                    
class System_Model():
    def __init__(self):
#        self.demand_obj = demand_df
        self.system_components = {} # Holds all system components
        self.system_hierarchy = {'0': [], '1': [], '2': []}
    
    def add_component(self, component, name, stage = None):
        if name in self.system_components:
            print(f'{name} already in system components. Either rename or remove existing component')
        else:
            if stage:
                self.system_hierarchy[stage].append(name)
            self.system_components[name] = component
            
    def remove_component(self, name):
        if name in self.system_components:
            del self.system_components[name]
        else:
            print(f'{name} is not in system')
         
    def simulate(self):
        def stage0():
            _output_vals = []
            for component in self.system_hierarchy['0']:
                if _output_vals == []:
                    _output_vals = self.system_components[component].demand
                else:
                    _output_vals = [x + y for x, y in zip(_output_vals, self.system_components[component].demand)]
                
            # Stage 1 operations are passive operations that occur to the system demand; e.g. passive ac-connected solar generation
            # output of Stage 1 is negative (demand) and positive (supply) of passive components
            return _output_vals
            
        def stage1(amt):
            # Stage 2 operations occur at the controller; e.g generator/battery demand response, arbitrage, etc.
            _output_vals = []
            for demand_vals in amt:
                _output_vals.append(self.system_components['cnt1'].io(demand_vals)) # Neg  values for demand
            
            return _output_vals
        
        def stage2(amt):
            #Stage 3 operations occur on as-yet unfulfilled load
            self.system_components['grd1'].supply(amt)
               
        demand = stage0()
        if 'cnt1' in self.system_components:
            demand = stage1(demand)
        stage2(demand)
#        self.simulated_df.index.names = ['Hour']
#        return sum(self.simulated_df['Demand'])
    
    def cost_sim(self, project_years, interest, inflation):
        "Calculate individual costs associated with each component"
        base_costs = 0
        yearly_costs = 0
        for component in self.system_components:
            if self.system_components[component].cost_component:
                base_costs = base_costs + self.system_components[component].cost_list['base_cost']
                yearly_costs = yearly_costs + self.system_components[component].cost_list['yearly_cost']
        base_costs = base_costs * (1+inflation)**project_years
        total_yearly_costs = []
        for year in range(project_years):
            total_yearly_costs.append(yearly_costs*(1+interest)**year)
        return base_costs + sum(total_yearly_costs)
        
    
    def clear(self):
        self.__init__()
        
    def __str__(self):
        
        components = ''
        for component in self.system_components:
            components = component + components
        return components

if __name__ == "__main__":
    a = solar.run_api(500, 500, 1)
    print(a.json_demand)
     
