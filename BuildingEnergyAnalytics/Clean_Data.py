'''

This file cleans a dataframe according to user specifications.

To Do
1. For remove_outliers() - may need a different boundary for each column.

Last modified: August 25 2018
@author Marco Pritoni <marco.pritoni@gmail.com>

'''

import numpy as np
import pandas as pd
from scipy import stats


class Clean_Data:

    def __init__(self, df):
        ''' Constructor '''
        self.original_data = df
        self.cleaned_data = pd.DataFrame()


    def drop_columns(self, col):
        ''' Drop columns in dataframe '''
        try:
            self.cleaned_data.drop(col, axis=1, inplace=True)
        except Exception as e:
            raise e


    def rename_columns(self, col):
        try:
            self.cleaned_data.columns = col
        except Exception as e:
            raise e


    def resample_data(self, data, freq):
        '''
            1. Also need to deal with energy quantities where resampling is .sum()
            2. Figure out how to apply different functions to different columns .apply()
            3. This theoretically work in upsampling too, check docs
            http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.resample.html 
        '''
        data = data.resample(freq).mean()
        return data


    def interpolate_data(self, data, limit, method):
        ''' Interpolate dataframe '''
        data = data.interpolate(how="index", limit=limit, method=method)        
        return data


    def remove_na(self, data, remove_na_how):
        ''' Remove NA from dataframe ''' 
        data = data.dropna(how=remove_na_how)        
        return data


    def remove_outliers(self, data, sd_val):
        '''
            This func removes all data data above or below n sd_val from the mean
            It also excludes all lines with NA in any column
        '''
        data = data.dropna()
        data = data[(np.abs(stats.zscore(data)) < float(sd_val)).all(axis=1)]
        return data


    def remove_out_of_bounds(self, data, low_bound, high_bound):
        ''' 
            Remove all points < low bound and > high bound 
            Add: A different boundary for each column
        '''
        data = data.dropna()
        data = data[(data > low_bound).all(axis=1) & (data < high_bound).all(axis=1)]        
        return data


    def clean_data(self, resample=True, freq='h', 
                    interpolate=True, limit=1, method='linear',
                    remove_na=True, remove_na_how='any', 
                    remove_outliers=True, sd_val=3, 
                    remove_out_of_bounds=True, low_bound=0, high_bound=9998):
        ''' Clean dataframe '''

        # Store copy of the original data
        data = self.original_data

        if resample:
            try:
                data = self.resample_data(data, freq)
            except Exception as e:
                raise e

        if interpolate:
            try:
                data = self.interpolate_data(data, limit=limit, method=method)
            except Exception as e:
                raise e
        
        if remove_na:
            try:
                data = self.remove_na(data, remove_na_how)
            except Exception as e:
                raise e

        if remove_outliers:
            try:
                data = self.remove_outliers(data, sd_val)
            except Exception as e:
                raise e

        if remove_out_of_bounds:
            try:
                data = self.remove_out_of_bounds(data, low_bound, high_bound)
            except Exception as e:
                raise e

        self.cleaned_data = data
