# -*- coding: utf-8 -*-
"""
@author : Armando Casillas <armcasillas@ucdavis.edu>
@author : Marco Pritoni <marco.pritoni@gmail.com>

Created on Wed Jul 26 2017
Update Aug 08 2017
Update Oct 25 2017

"""
from __future__ import division
import pandas as pd
import os
import sys
import requests as req
import json
import numpy as np
import datetime
import pytz
from pandas import rolling_median
from matplotlib import style
import matplotlib


class ts_util(object):

######################################################################## 
## simple load file section - eventually replace this with CSV_Importer 

    def _set_TS_index(self, data):
        '''
        This internal method transforms the index into a datetime index and the values into float 

        Parameters
        ----------
        data : DataFrame or Series

        Returns
        -------
        data : DataFrame or Series
            same data as in input with type changed
        '''
        # set index
        data.index = pd.to_datetime(data.index)

        # format types to numeric
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        return data

    def load_TS(self, fileName, folder):
        '''      
        This method loads a single .csv into a Dataframe.
        It sets the first column as datetime index and all the rest of the data as float
        For more advanced option use other classes created to load csv or get data from APIs

        Parameters
        ----------
        fileName : str
            filename of the .csv file to be loaded
        folder : str
            folder containing the file

        Returns
        -------
        data : DataFrame
            DataFrame with time-series data
        '''
        path = os.path.join(folder, fileName)
        data = pd.read_csv(path, index_col=0)
        data = self._set_TS_index(data)
        return data

######################################################################## 
## time correction for time zones - eventually replace this with CSV_Importer 

    def _utc_to_local(self, data, local_zone="America/Los_Angeles"):
        '''
        This method takes in pandas DataFrame and adjusts index according to timezone in which is requested by user

        Parameters
        ----------
        data: Dataframe
            pandas dataframe of timeseries data

        local_zone: str, default "America/Los_Angeles"
            pytz.timezone string of specified local timezone to change index to

        Returns
        -------
        data: Dataframe
            Pandas dataframe with timestamp index adjusted for local timezone
        '''
        data.index = data.index.tz_localize(pytz.utc).tz_convert(
            local_zone)  # accounts for localtime shift
        # Gets rid of extra offset information so can compare with csv data
        data.index = data.index.tz_localize(None)

        return data

    def _local_to_utc(self, timestamp, local_zone="America/Los_Angeles"):
        '''      
        This method transforms a string in local time to a string in UTC time.
        Most servers hold the data in UTC time therfore an API call has to request the data in that format

        Parameters
        ----------
        timestamp: str
            string of time in local time

        local_zone: str, default "America/Los_Angeles"
            string with timezone name according to pandas pytz convention

        Returns
        -------

        timestamp_new: str
            timestamp string in ISO format (eg: "2017-12-30 22:30:40") in utc time
        '''

        timestamp_new = pd.to_datetime(
            timestamp, infer_datetime_format=True, errors='coerce')

        timestamp_new = timestamp_new.tz_localize(
            local_zone).tz_convert(pytz.utc)

        timestamp_new = timestamp_new.strftime('%Y-%m-%d %H:%M:%S')

        return timestamp_new

######################################################################## 
## remove start and end NaN

    
    def first_valid_per_col (self,data):

        return data.apply(lambda col: col.first_valid_index())


    def last_valid_per_col (self,data):

        return data.apply(lambda col: col.last_valid_index())


    def remove_start_NaN(self, data, how="all"):

        '''
        This method removes the heading NaN

        Parameters
        ----------
        data : DataFrame or Series
            input raw data
        how : str, default "any"
            "any", if any column has missing data the whole row is removed, 
            "all", all columns have to have missing data for the row to be removed

        Returns
        -------
            data:  DataFrame or Series 
                data with starting missing data at the beginning removed 
        '''
        try:
            idx = data.dropna(how=how).index[0] #find the first dropped value in all or any columns with missing data 
              
            data.loc[idx:]
        except:
            data = pd.DataFrame()
        
        return data 

    def remove_end_NaN(self, data, how="all"):

        '''      
        This method removes the tailing NaN

        Parameters
        ----------
        data : DataFrame or Series
            input raw data
        how : str, default "any"
            "any", if any column has missing data the whole row is removed, 
            "all", all columns have to have missing data for the row to be removed

        Returns
        -------
            data:  DataFrame or Series 
                data with ending missing data at the beginning removed 
        '''
        try:
            idx = data.dropna(how=how).index[-1] #find the last dropped value in all or any columns with missing data 
            data.loc[:idx]
        except:
            data = pd.DataFrame()

        return  data 


######################################################################## 
## Missing data section

    def _find_missing(self, data, return_bool=True, how="any"):

        '''
        This method takes in  DataFrame or Series and find missing values in each column
        It returns either the boolean_selector (Series) of missing data rows (with return_bool = "any" or "all")
        or the DataFrame/Series of True where data is missing (with return_bool = False)

        Parameters
        ----------
        data : DataFrame or Series
            input timeseries data
        return_bool : bool, default True
            if True it returns a boolean selector (Series), depending on how paramenters
        how = str, default "any"
            "any", if any column has missing data it returns True for the row
            "all", all columns have to have missing data for the row to True
        
        Returns
        -------
        data or bool_sel: DataFrame or Series
            Dataframe or Series with True where data is missing (return_bool = False) 
            or 
            boolean selector Series with True where "any" or "all" cols are null

        '''
        
        if isinstance(data, pd.Series): # if data is Series "any" or "all" does not apply

            bool_sel = data.isnull() # this returns the full Series with True where the condition is true

            return bool_sel # Series

        elif (isinstance(data, pd.DataFrame)) & (return_bool == False): # returns the whole DataFrame with True where the condition is true in each column

            data = data.isnull()

            return data #DataFrame

        elif (isinstance(data, pd.DataFrame)) & (return_bool == True):

            if how == "any": # this returns a bool selector if any of the condition on columns is True

                bool_sel = data.isnull().any(axis=1)

                return bool_sel # Series

            elif how == "all": # this returns a bool selector if all of the coditions on columns are True

                bool_sel = data.isnull().all(axis=1)

                return bool_sel #Series

        else:
            print("error in return_bool input")

        return pd.Series()

    def display_missing(self, data, how="any"):

        '''
        This method takes in DataFrame or Series and returns missing values in each column
        It returns a DataFrame/Series of missing data rows

        Parameters
        ----------
        data : DataFrame or Series
            input timeseries data
        how = str, default "any"
            "any", if any column has missing data it returns True for the row
            "all", all columns have to have missing data for the row to True
        
        Returns
        -------
        data: DataFrame or Series
            Dataframe with missing data only 

        '''

        bool_sel = self._find_missing(data,return_bool=True, how=how)

        return data[bool_sel]

    def count_missing(self, data, output="number"):

        '''
        This method takes in DataFrame or Series and returns the count of missing values in each column
        It returns number or percent of missing rows

        Parameters
        ----------
        data : DataFrame or Series
            input timeseries data
        output = str, default "number"
            number: count of rows
            output: % of rows
        
        Returns
        -------
        ret: Series with count (number or percentage) of missing value for each column (column name in Series index) 
        '''

        count = pd.DataFrame(self._find_missing(data,return_bool=False)).sum()

        if output == "number":

            return count

        elif output == "percent":

            return ((count / (data.shape[0])) * 100)

    def remove_missing(self, data, return_bool=True, how="any"):

        '''
        This method takes in DataFrame or Series and removes missing values on any or all columns

        Parameters
        ----------
        data : DataFrame or Series
            input timeseries data
        return_bool : bool, default True
            if True it returns a boolean selector (Series), depending on how paramenters
        how = str, default "any"
            "any" removes rows if any column has missing data 
            "all" removes rows if all columns have to have missing data
        
        Returns
        -------
        data: DataFrame or Series
            original data with missing data removed

        '''
        '''      
        Parameters
        ----------
        data: Dataframe
        output : "number" or "percent"

        Returns
        -------
        DataFrame or Series without missing data. With "any" all the cols have at least one null value are removed, with "all" only cols with all null 
        values are removed.

        '''

        bool_sel = self._find_missing(data,return_bool=True, how=how)

        return data[~bool_sel]


######################################################################## 
## If condition section

    def _find_equal_to_values_return_frame(self, data, val):

        '''   
        This method returns the  
        Parameters
        ----------

        Returns
        -------
        '''       
        if isinstance(val, pd.Series):
            
            if val.index[0] in data.columns:
        
                data_select = pd.DataFrame(data.eq(val, axis=1))
        
            elif val.index[0] in data.index:

                data_select = pd.DataFrame(data.eq(val, axis=0))

            else:

                data_select = pd.DataFrame()
                print("mismatch between val and data")
    
        else:
        
            data_select = pd.DataFrame(data.eq(val))

        return data_select


    def _find_greater_than_values_return_frame(self, data, val):

        '''      
        Parameters
        ----------

        Returns
        -------
        '''                    
        if isinstance(val, pd.Series):
            
            if val.index[0] in data.columns:
        
                data_select = pd.DataFrame(data.gt(val, axis=1))
        
            elif val.index[0] in data.index:

                data_select = pd.DataFrame(data.gt(val, axis=0))

            else:

                data_select = pd.DataFrame()
                print("mismatch between val and data")
    
        else:
        
            data_select = pd.DataFrame(data.gt(val))

        return data_select

    
    def _find_less_than_values_return_frame(self, data, val):
 
        '''      
        Parameters
        ----------

        Returns
        -------
        '''                    
            
        if isinstance(val, pd.Series):
            
            if val.index[0] in data.columns:
        
                data_select = pd.DataFrame(data.lt(val, axis=1))
        
            elif val.index[0] in data.index:

                data_select = pd.DataFrame(data.lt(val, axis=0))

            else:

                data_select = pd.DataFrame()
                print("mismatch between val and data")
    
        else:
        
            data_select = pd.DataFrame(data.lt(val))

        return data_select


    def _find_greater_than_or_equal_to_values_return_frame(self, data, val):

        '''      
        Parameters
        ----------

        Returns
        -------
        '''           
            
        if isinstance(val, pd.Series):
            
            if val.index[0] in data.columns:
        
                data_select = pd.DataFrame(data.ge(val, axis=1))
        
            elif val.index[0] in data.index:

                data_select = pd.DataFrame(data.ge(val, axis=0))

            else:

                data_select = pd.DataFrame()
                print("mismatch between val and data")
    
        else:
        
            data_select = pd.DataFrame(data.ge(val))

        return data_select

    def _find_less_than_or_equal_to_values_return_frame(self, data, val):

        '''      
        Parameters
        ----------

        Returns
        -------
        ''' 
            
        if isinstance(val, pd.Series):
            
            if val.index[0] in data.columns:
        
                data_select = pd.DataFrame(data.le(val, axis=1))
        
            elif val.index[0] in data.index:

                data_select = pd.DataFrame(data.le(val, axis=0))

            else:

                data_select = pd.DataFrame()
                print("mismatch between val and data")
    
        else:
        
            data_select = pd.DataFrame(data.le(val))

        return data_select

    def _find_different_from_values_return_frame(self, data, val):

        '''      
        Parameters
        ----------

        Returns
        -------
        ''' 
        if isinstance(val, pd.Series):
            
            if val.index[0] in data.columns:
        
                data_select = pd.DataFrame(data.ne(val, axis=1))
        
            elif val.index[0] in data.index:

                data_select = pd.DataFrame(data.ne(val, axis=0))

            else:

                data_select = pd.DataFrame()
                print("mismatch between val and data")
   
        else:
        
            data_select = pd.DataFrame(data.ne(val))

        return data_select

    def _find_if(self, data, operator, val, return_bool="any"):
        
        '''
        This method takes in  DataFrame or Series and find rows that match if + operator + val 
        It returns either the boolean_selector of rows that match the condition (with return_bool = "any" or "all")
        or the DataFrame/Series of True where data matches condition

        Parameters
        ----------
        data : DataFrame or Series
        operator :  "=", ">" ... etc
        val : float or int to compare to
        return_bool : "any", "all" or False
        
        Returns
        -------
        Dataframe with True where data matches condition (return_bool = False) 
        or 
        boolean selector Series with True where "any" or "all" cols match condition

        '''

        if operator == "=":
        
            data = self._find_equal_to_values_return_frame(data,val)

        elif operator == ">":

            data = self._find_greater_than_values_return_frame(data,val)
      
        elif operator == "<":

            data = self._find_less_than_values_return_frame(data,val)

        elif operator == ">=":

            data = self._find_greater_than_or_equal_to_values_return_frame(data,val)

        elif operator == "<=":

            data = self._find_less_than_or_equal_to_values_return_frame(data,val)
       
        elif operator == "!=":

            data = self._find_different_from_values_return_frame(data,val)


        if return_bool == False: # this returns the full table with True where the condition is true

            return data

        elif return_bool == "any": # this returns a bool selector if any of the condition on columns is True

            bool_sel = data.any(axis=1)

            return bool_sel

        elif return_bool == "all": # this returns a bool selector if all of the coditions on columns are True

            bool_sel = data.all(axis=1)

            return bool_sel

        else:
            print("error occurred")

        return pd.Series()

    def display_if(self, data, operator, val, return_bool="any"):

        '''      
        Parameters
        ----------
        data: Dataframe or Series
        operator :  "=", ">" ... etc
        val : float or int to compare to
        return_bool : "any", "all" or False

        Returns
        -------
        DataFrame or Series with data matching the condition. 
        With "any" all the cols have at least one matching col, with "all" all cols need to match

        '''

        bool_sel = self._find_if(data, operator, val, return_bool=return_bool)

        return data[bool_sel]

    def count_if(self, data, operator, val, output="number"):

        '''
        to write
        '''

        count = self._find_if(data, operator, val, return_bool=False).sum()


        if output == "number":

            return count
    
        elif output == "percent":
        
            return count/data.shape[0]*1.0*100

        return count

    def remove_if(self, data, operator, val, return_bool="any"):

        '''
        to write
        '''

        bool_sel = self._find_if(data, operator, val, return_bool=return_bool)

        return data[~bool_sel]

######################################################################## 
## Out of Bound section

    def _find_outOfBound(self, data, lowBound, highBound, return_bool=False):

        '''      
        Parameters
        ----------

        Returns
        -------
        '''
        
        # For return_bool=False, tmp's type is a dataframe
        # For return_bool='any' or 'all', tmp's type is a series 
        tmp = (self._find_if(data, "<", lowBound, return_bool)) |  (self._find_if(data, ">", highBound, return_bool))
        
        if return_bool == False: # this returns the full table with True where the condition is true

            return tmp

        elif return_bool == "any": # this returns a bool selector if any of the condition on columns is True

            tmp.index = data.index
            return pd.DataFrame(tmp)

        elif return_bool == "all": # this returns a bool selector if all of the conditions on columns are True

            tmp.index = data.index
            return pd.DataFrame(tmp)

        else:
            print("error occurred")


        return pd.Series()

    def display_outOfBound(self, data, lowBound, highBound, return_bool="any"):

        '''      
        Parameters
        ----------

        Returns
        -------
        '''

        bool_sel = self._find_outOfBound(data, lowBound, highBound, return_bool=return_bool)
        #print bool_sel[0]
        
        return data[bool_sel[0]]

    def count_outOfBound(self, data, lowBound, highBound, output="number"):

        '''      
        Parameters
        ----------

        Returns
        -------
        '''

        count = self._find_outOfBound(data, lowBound, highBound, return_bool=False).sum()


        if output == "number":

            return count
    
        elif output == "percent":
        
            return count/data.shape[0]*1.0*100

        return count

    def remove_outOfBound(self, data, lowBound, highBound, return_bool="any"):

        '''      
        Parameters
        ----------

        Returns
        -------
        '''

        bool_sel = self._find_outOfBound(data, lowBound, highBound, return_bool=return_bool)

        return data[~bool_sel[0]]

######################################################################## 
## Outliers section

    def _calc_outliers_bounds(self, data, method, coeff, window):
        '''      
        Parameters
        ----------

        Returns
        -------
        '''
        if method == "std":

            lowBound = (data.mean(axis=0) - coeff * data.std(axis=0))#.values[0]
            highBound = (data.mean(axis=0) + coeff * data.std(axis=0))#.values[0]

        elif method == "rstd":

       	    rl_mean=data.rolling(window=window).mean(how=any)
            rl_std = data.rolling(window=window).std(how=any).fillna(method='bfill').fillna(method='ffill')

            lowBound = rl_mean - coeff * rl_std

            highBound = rl_mean + coeff * rl_std

        elif method == "rmedian":

            rl_med = data.rolling(window=window, center=True).median().fillna(
                method='bfill').fillna(method='ffill')

            lowBound =  rl_med - coeff
            highBound = rl_med + coeff

        elif method == "iqr":         # coeff is multip for std and IQR or threshold for rolling median

            Q1 = data.quantile(.25)     # coeff is multip for std or % of quartile
            Q3 = data.quantile(.75)
            IQR = Q3 - Q1

            lowBound = Q1 - coeff * IQR
            highBound = Q3 + coeff * IQR

        elif method == "qtl":

            lowBound = data.quantile(.005)
            highBound = data.quantile(.995)

        # New method added: Anything above/below 2*mean is an outlier
        # TO DO: change calc when mean = 0
        elif method == "new":
            
            lowBound = (data.mean(axis=0) - data.mean(axis=0)).values[0]
            highBound = (data.mean(axis=0) + data.mean(axis=0)).values[0]

        else:
            print ("method chosen does not exist")
            lowBound = None
            highBound = None


        return lowBound, highBound

    def _find_outliers(self, data, method="std", coeff=3, window=10):
        '''      
        Parameters
        ----------

        Returns
        -------
        '''
        lowBound, highBound = self._calc_outliers_bounds(
            data, method, coeff, window)

        return data

    def display_outliers(self, data, method, coeff, window=10):
        '''      
        Parameters
        ----------

        Returns
        -------
        '''
        lowBound, highBound = self._calc_outliers_bounds(
            data, method, coeff, window)

        data = self.display_outOfBound(data, lowBound, highBound)

        return data

    def count_outliers(self, data, method, coeff, output, window=10):
        '''      
        Parameters
        ----------

        Returns
        -------
        '''
        lowBound, highBound = self._calc_outliers_bounds(
            data, method, coeff, window)

        count = self.count_outOfBound(data, lowBound, highBound, output=output)
        
        

        return count

    def remove_outliers(self, data, method, coeff, window=10):
        '''      
        Parameters
        ----------

        Returns
        -------
        '''
        lowBound, highBound = self._calc_outliers_bounds(
            data, method, coeff, window)

        data = self.remove_outOfBound(data, lowBound, highBound)

        return data



######################################################################## 
## Missing Data Events section

    def get_start_events(self, data, var = "T_ctrl [oF]"): # create list of start events
        '''      
        Parameters
        ----------

        Returns
        -------
        '''
        start_event = (data[var].isnull()) & ~(data[var].shift().isnull()) # find NaN start event
        start = data[start_event].index.tolist() # selector for these events


        if np.isnan(data.loc[data.index[0],var]): # if the first record is NaN
            start =  [data.index[0]] + start # add first record as starting time for first NaN event
        else:
            start = start
        return start


    def get_end_events(self, data, var = "T_ctrl [oF]"): # create list of end events
        '''      
        Parameters
        ----------

        Returns
        -------
        '''
        end_events = ~(data[var].isnull()) & (data[var].shift().isnull()) # find NaN end events
        end = data[end_events].index.tolist() # selector for these events

        if ~np.isnan(data.loc[data.index[0],var]): # if first record is not NaN
            end.remove(end[0]) # remove the endpoint ()

        if np.isnan(data.loc[data.index[-1],var]): # if the last record is NaN
            end =  end + [data.index[-1]]  # add last record as ending time for first NaN event
        else:
            end = end

        return end


    def create_event_table(self, data, var): # create dataframe of of start-end-length for current house/tstat
        '''      
        Parameters
        ----------

        Returns
        -------
        '''    
        # remove initial and final missing data
        self.remove_start_NaN(data, var)
        self.remove_end_NaN(data, var)
        
        # create list of start events
        start = self.get_start_events(data, var)
        
        # create list of end events
        end = self.get_end_events(data, var)
            
        # merge lists into dataframe and calc length
        events = pd.DataFrame.from_items([("start",start), ("end",end )])
        
        events["length_min"] = (events["end"] - events["start"]).dt.total_seconds()/60 # note: this needs datetime index
        
        #print events
        events.set_index("start",inplace=True)

        return events

        