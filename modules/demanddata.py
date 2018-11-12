# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 11:13:42 2018

@author: BJoseph
"""

import pandas as pd

class import_data:
    def __init__(self, df):
        '''
        Return pandas dataframe object for a facility's electricity demand
        '''
        self.df = df

    @classmethod
    def fifteenMinute(cls, file):
        '''
        Import csv file and convert data to hourly demand data if 15 minute 
        intervals. Convert to pd dataframe
        '''
        
        # Import from CSV
        df = pd.read_csv(file, names=['Demand'])
        
        #Convert to Watts
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
        return cls(df)
    
    @classmethod
    def hourlyInterval(cls, file):
        '''
        Import csv file and convert to pd dataframe
        '''
        df = pd.read_csv(file, names=['Demand'])
        df.index.names=['Hour']
        
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
        elif len(df) >8760:
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

if __name__ == "__main__":
    file = 'dc_foods_2014.csv'
    data = import_data.fifteenMinute(file)
