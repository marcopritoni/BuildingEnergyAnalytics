# -*- coding: utf-8 -*-
"""
@author : Marco Pritoni <marco.pritoni@gmail.com>

"""
import pandas as pd
import numpy as np
import datetime
import pytz
import os


class ts_util(object):
    """
    This class offers several helper functions to find, display, count, remove and replace rows with that match 
    specific conditions in timeseries. 
    
    Applications include count or remove outliers to prepare the data for modeling.
    """

########################################################################     
## simple load file section - eventually replace this with CSV_Importer 

    def _set_TS_index(self, data):
        """
        This internal method transforms the index into a datetime index and the values into float 

        Parameters
        ----------
        data : DataFrame or Series

        Returns
        -------
        data : DataFrame or Series
            same data as in input with type changed
        """
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


######################################################################## ######################################################################## 

    def _find(self, data, func, return_bool, how, *args, **kargs):
        """
        This method applies a passed function (eg: outliers) to a Series or DataFrame.
        If Series is passed as input, the output is a Bool Selector.
        If DataFrame is passed as input, the output depends on return_bool arg. 
        If return_bool is False the output is a DataFrame of Bool Selectors (note this CANNOT be used as mask)
        If return_bool is True, the output is a Bool Selector Series with True if "any" or "all" the cols are True,
        depending on the how arg.

        Parameters
        ----------
        data : DataFrame or Series of timeseries data
        func : pointer to a function
        return_bool : boolean
        how : str
        *args, **kargs : additional args for the passed func
        
        Returns
        -------
        df_bool_sel or bool_sel : DataFrame or Series

        """
        
        if isinstance(data, pd.Series): # if data is Series "any" or "all" does not apply

            bool_sel = func(data, *args, **kargs) # this returns a Series of True where the condition is true

            return bool_sel # Series

        elif (isinstance(data, pd.DataFrame)):
            
            try:
                df_bool_sel = func(data, *args, **kargs) # this returns a DataFrame of True where the condition is true
                
            except:
                
                df_bool_sel = data.apply(func, *args, **kargs) # TODO: NEED TO CHANGE THIS ONE !!! 
                ## TODO: Note different func have different behavior here!!

            if (return_bool == False): # returns the whole DataFrame with True where the condition is true in each column

                return df_bool_sel #DataFrame

            elif (return_bool == True):

                if how == "any": # this returns a bool selector if any of the condition on columns is True

                    return df_bool_sel.any(axis=1) # Series

                elif how == "all": # this returns a bool selector if all of the coditions on columns are True

                    return df_bool_sel.all(axis=1) # Series

        else:
            print("error in data type")

        return pd.Series() # Series


    def _display(self, data, func, how="any", mask_others=True, *args, **kargs):
        """
        This method applies a passed function (eg, outliers) to a Series or DataFrame and returns the original
        object with a selection of rows where the condition specified by the func is True (eg, non outliers values).
        If a DataFrame is provided as input, user can select to see the rows that have any or all the columns
        matching the condition.
        Update: mask_others == True replaces the data not displayed with NaN, in case of multi-columns DataFrame
        and how == "any" selected
        
        Example
        -------
        >>> data
                    sensor1  sensor2
        2017-12-01     73.0      NaN
        2017-12-02     74.0     70.0
        2017-12-03      NaN     60.0
        2017-12-04     82.0     55.0
        2017-12-05      NaN     50.0
        
        >>> _display(data, _function_if, how="any", mask_others=True, operator=">", val=60)

                   sensor1  sensor2
        2017-12-01     73.0      NaN
        2017-12-02     74.0     70.0
        2017-12-04     82.0      NaN <- mask_others=True transforms this into NaN, beacause it is not > 60
        
        display_missing sets mask_others == False to avoid confusion with the NaN displayed on purpose
        
        Parameters
        ----------
        data : DataFrame or Series timeseries data
        func : pointer to a function
        how = str, default "any"
            "any", returns True if any column matches the codition tested by the func
            "all", returns True if all columns match the codition tested by the func
        
        *args, **kargs : additional args for the passed func
        
        Returns
        -------
        data: DataFrame or Series
        """
        invert_selection = {"any":"all", "all":"any"} # this is used because dropna() and _find use how in opposite ways
        
        if mask_others:
            gen_bool_sel = self._find(data, func, return_bool=False, how=how, *args, **kargs) # bool_sel if data is Series; df_bool_sel if data is DataFrame
            data = data.mask(~gen_bool_sel).dropna(how=invert_selection[how])
        else:
            bool_sel = self._find(data, func, return_bool=True, how=how, *args, **kargs)
            data = data[bool_sel]

        return data

    
    def _remove(self, data, func, return_bool=True, how="any", mask_others=True, *args, **kargs):
        """
        This method takes in DataFrame or Series and removes the rows that match the criteria
        specified by the func provided.
        If a DataFrame is provided as input, user can select to remove the rows that have any or all the columns
        matching the condition.
        Update: mask_others == True replaces the data not removed with NaN, in case of multi-columns DataFrame
        and how == "all" selected
        
        Example
        -------
        >>> data
                    sensor1  sensor2
        2017-12-01     73.0      NaN
        2017-12-02     74.0     70.0
        2017-12-03      NaN     60.0 
        2017-12-04     82.0     55.0
        2017-12-05      NaN     50.0
        2017-12-06      NaN      NaN 
        
        >>> _remove(data, _function_if, how="all", mask_others=True, operator="<", val=50)

                    sensor1  sensor2
        2017-12-02     74.0     70.0
        2017-12-04     82.0     55.0 
        
        This removes rows that have all cols < 50 or NaN.
        
        display_missing sets mask_others == False to avoid confusion with the NaN displayed on purpose
        

        Parameters
        ----------
        data : DataFrame or Series of timeseries data
        func : pointer to a function
        return_bool : bool, default True
            if True it returns a boolean selector (Series), depending on how paramenters
        how = str, default "any"
            "any", returns True if any column matches the codition tested by the func
            "all", returns True if all columns match the codition tested by the func
        *args, **kargs : additional args for the passed func
        
        Returns
        -------
        data: DataFrame or Series
            original data with missing data removed

        """
        if mask_others:
            gen_bool_sel = self._find(data, func, return_bool=False, how=how, *args, **kargs) # bool_sel if data is Series; df_bool_sel if data is DataFrame
            data = data.mask(gen_bool_sel).dropna(how=invert_selection[how])
        else:
            bool_sel = self._find(data, func, return_bool=True, how=how, *args, **kargs)
            data = data[~bool_sel]
        
        bool_sel = self._find(data, func , return_bool=return_bool, how=how, *args, **kargs)
        
        ## TODO: remove for each column and replace with NaN ??
        
        return data[~bool_sel]
        
        
    def _define_output(self, count, count_all, output):
        """
        Used by _count method to specify the output: number or percent

        Parameters
        ----------
        count : Series
        count_all : Series 
        output : str
        
        Returns
        -------
        Series
        """
        
        if output == "number":

            return count

        elif output == "percent":
            
            return ((count / count_all) * 100)


    def _count(self, data, func, output="number", *args, **kargs):
        """
        This method takes in DataFrame or Series and returns the count (# or %) of rows that match the criteria
        specified by the func provided.

        Parameters
        ----------
        data : DataFrame or Series of timeseries data
        func : pointer to a function
        output = str, default "number"
            number: count of rows
            percent: % of rows
        *args, **kargs : additional args for the passed func
        
        Returns
        -------
        ret: Series
        """

        count = pd.DataFrame(self._find(data, func, return_bool=False, how="any", *args, **kargs)).sum()
        count_all = data.shape[0]

        return self._define_output(count=count, count_all=count_all, output=output)

    
    def _replace(self,data):
        """
        TODO: add function to replace data removed with a) fixed value b) something provided by func2 
        c) something provided by a model
        """
        
        return
        
########################################################################################################
## BASIC FUNCTIONS 

    def _func_missing(self, data): 
        """
        This method returns True for each element that is NaN.
        Same behavior for Series and DataFrame.
        
        Parameters
        ----------
        data : DataFrame or Series of timeseries data
        
        Returns
        -------
        DataFrame or Series        
        
        """
        return data.isnull()

    
    def _func_flatlines(self, data): 
        """
        This method returns True for periods with constant values (that do not change in time).
        Same behavior for Series and DataFrame.
        
        Parameters
        ----------
        data : DataFrame or Series of timeseries data
        
        Returns
        -------
        DataFrame or Series        
        
        """    
        return (((data.diff(-1)==0) | (data.diff(+1)==0)))
    
    
    def _infer_axis(self, data, val):
        """
        This helper method is used by _func_if to infer if the Series provided for comparison (val) 
        references column names or the timeseries index. This impacts the selction of the arg axis to
        pass to _flex_comp_func.
        
        Parameters
        ----------
        data : DataFrame or Series of timeseries data
        val: Series
        
        Returns
        -------
        int 
            axis to be passed 
        
        """
        if isinstance(val, pd.Series): # this check if val is a Series not data
            
            if val.index.equals(data.index): # a time series is provided as value
                
                return 0 # axis = 0
        
            elif val.index.equals(data.columns): # a set of columns is provided as value
        
                return 1 # axis =1
            
        else:                          # either a full DataFrame or a single value
            return 1 # axis =1

            
    def _flex_comp_func(self, data, operator, val, axis):        
        """
        This helper method takes in an operator (eg, "<=") and calls the flexible comparison method of
        a DataFrame or Series and apply it to the data provided. The result is a DataFrame or Series with
        True where the condition is matched.
        Same behavior for Series and DataFrame inputs.

        Parameters
        ----------
        data : DataFrame or Series of timeseries data
        operator: str
            "=", "==", ">", "<", ">=", "<=", "!=", "~"
        val: int, float, Series or DataFrame
                
        Returns
        -------
        DataFrame or Series
            
        
        """
        
        if (operator == "=") or (operator == "=="):
        
            return data.eq(val, axis=axis)

        elif operator == ">":

            return data.gt(val, axis=axis)
      
        elif operator == "<":

            return data.lt(val, axis=axis)

        elif operator == ">=":

            return data.ge(val, axis=axis)

        elif operator == "<=":

            return data.le(val, axis=axis)
       
        elif (operator == "!=") or (operator == "~"):

            return data.ne(val, axis=axis)

        
    def _func_if(self, data, operator, val):

        """
        This method takes in an operator (eg, "<=") and a val and returns a DataFrame or Series (depending on data)
        with True where the condition is true.
        val can be:
        - a single value (eg, 74), 
        - a timeseries Series with the same index of the data (eg, {"2018-01-01" : 74, "2018-01-02" : 79 ...})
        - a Series with a single value for each column (eg, "sensor1": 100, "sensor2":80))
        - a DataFrame with the same shape, index and columns of the data
        
        Parameters
        ----------
        data : DataFrame or Series of timeseries data
        operator: str
            "=", "==", ">", "<", ">=", "<=", "!=", "~"
        val: int, float, Series or DataFrame
                
        Returns
        -------
        DataFrame or Series
            
        Examples
        --------
        >>> data = pd.DataFrame(index= ["2017-12-01", "2017-12-02","2017-12-03", "2017-12-04", "2017-12-05"], 
                    data = {"sensor1": [73,74,81,82,90], "sensor2": [90,70,60,55,50]})
        >>> print(data)
                    sensor1  sensor2
        2017-12-01       73       90
        2017-12-02       74       70
        2017-12-03       81       60
        2017-12-04       82       55
        2017-12-05       90       50
        
        >>> val=80 # single value
        >>> ts._func_if(data,">=",val)
        
                    sensor1  sensor2
        2017-12-01    False     True
        2017-12-02    False    False
        2017-12-03     True    False
        2017-12-04     True    False
        2017-12-05     True    False
        
        >>> val=pd.Series(index=["2017-12-01", "2017-12-02","2017-12-03", "2017-12-04", "2017-12-05"],
                    data= [73,74,81,82,90]) # timeseries Series
        >>> print(val)

        2017-12-01    73
        2017-12-02    74
        2017-12-03    81
        2017-12-04    82
        2017-12-05    90
        dtype: int64
        
        >>> ts._func_if(data,">=",val)
        
                    sensor1  sensor2
        2017-12-01     True     True
        2017-12-02     True    False
        2017-12-03     True    False
        2017-12-04     True    False
        2017-12-05     True    False

        >>> val=pd.Series(index=["sensor1","sensor2"],data= [50,100]) # Series with different value for each column
        >>> print(val)
        
        sensor1     50
        sensor2    100
        dtype: int64
        
        >>> ts._func_if(data,">=",val)
        
                    sensor1  sensor2
        2017-12-01     True    False
        2017-12-02     True    False
        2017-12-03     True    False
        2017-12-04     True    False
        2017-12-05     True    False
        
        >>> val=data = pd.DataFrame(index= ["2017-12-01", "2017-12-02","2017-12-03", "2017-12-04", "2017-12-05"], 
                            data = {"sensor1": [73,74,81,82,90], "sensor2": [90,70,60,55,50]})
        >>> print(val)        
        
            sensor1  sensor2
        2017-12-01       73       90
        2017-12-02       74       70
        2017-12-03       81       60
        2017-12-04       82       55
        2017-12-05       90       50        
        
        >>> ts._func_if(data,">=",val)
        
                    sensor1  sensor2
        2017-12-01     True     True
        2017-12-02     True     True
        2017-12-03     True     True
        2017-12-04     True     True
        2017-12-05     True     True
        """

        if isinstance(data, pd.Series):
            
            if isinstance(val, pd.DataFrame):
                
                print ("mismatch data and values to compare agaist")
                
                return 
            
            return self._flex_comp_func(data=data, operator=operator, val=val, axis=0) # 
            
        elif isinstance(data, pd.DataFrame):
            
            val_axis = self._infer_axis(data, val)
        
            return self._flex_comp_func(data=data, operator=operator, val=val, axis=val_axis)
    
    
    def _func_outOfBound(self, data, lowBound, highBound, include_boundaries=False):
        """
        This method takes in a DataFrame or Series and calculates if it is outside of high or low bound.
        The lowBound and highBound can be:
        - a single value (eg, 74), 
        - a timeseries Series with the same index of the data (eg, {"2018-01-01" : 74, "2018-01-02" : 79 ...})
        - a Series with a single value for each column (eg, "sensor1": 100, "sensor2":80))
        - a DataFrame with the same shape, index and columns of the data
        
        Note: currently there is no check that lowBound < highBound
        
        Parameters
        ----------
        data : DataFrame or Series of timeseries data
        lowBound: int, float, Series or DataFrame
        highBound: int, float, Series or DataFrame
        include_boundaries : bool
            If False it checks for values strictly out of the bounds
        
        Returns
        -------
        DataFrame or Series        
        """
        if include_boundaries:
            return((self._func_if(data, "<=", lowBound)) |  (self._func_if(data, ">=", highBound)))
        else:
            return((self._func_if(data, "<", lowBound)) |  (self._func_if(data, ">", highBound)))


    def _calc_outliers_bounds(self, data, method, coeff, window, qcoeff, quant):
        """
        This helper method is used by _func_outliers to find the low and high bounds using different methods. 
        
        Add new methods here.

        Parameters
        ----------
        data : DataFrame or Series of timeseries data
        method : str
            "std" : bounds are  +- coeff * std_dev around the mean of each time series (constant in time)
            "iqr" : bounds are  +- coeff * interquartile range (constant in time)
            "qtl": bounds are outside of a specific quantile (const in time)
            "rstd" : bounds are  +- coeff * std_dev around the mean of each time series in a window of n timestamps (time-varying)
            "rmedian" : same as above but around the meadian
        coeff : int
            coeff to be multiplied to std dev 
        window : int
            window size (n rows) for rolling variants
        qcoeff : float or int
            coeff to be multiplied to iqr
        quant : float < 1 
            quantile
        
        Returns
        -------
        Two DataFrames if data input is a DataFrame and method produces time-varying bounds
        Two timeseries Series if data input is a Series and method produces time-varying bounds - or - 
        if data input is a DataFrame and method produces constant bounds
        
        """        
        if method == "std":

            lowBound = (data.mean(axis=0) - coeff * data.std(axis=0))
            highBound = (data.mean(axis=0) + coeff * data.std(axis=0))   

        elif method == "iqr":         #TODO: add to args diff coeff for this one, default 1.5

            Q1 = data.quantile(.25) # first quartile  
            Q3 = data.quantile(.75) # third quartile  
            IQR = Q3 - Q1 #interquartile range

            lowBound = Q1 - qcoeff * IQR # typically  coeff = 1.5
            highBound = Q3 + qcoeff * IQR # typically  coeff = 1.5

        elif method == "qtl":
            
             #TODO: add to args

            lowBound = data.quantile(quant)
            highBound = data.quantile(1-quant)
        
        elif method == "rstd":

       	    rl_mean=data.rolling(window=window).mean()
            rl_std = data.rolling(window=window).std().fillna(method='bfill').fillna(method='ffill')

            lowBound = rl_mean - coeff * rl_std
            highBound = rl_mean + coeff * rl_std

        elif method == "rmedian":

            rl_med = data.rolling(window=window, center=True).median().fillna(
                method='bfill').fillna(method='ffill')

            lowBound =  rl_med - coeff
            highBound = rl_med + coeff

        else:
            print ("method chosen does not exist")
            lowBound = None
            highBound = None

        return lowBound, highBound
    
    
    def _func_outliers(self, data, method="std", coeff=3, window=10, qcoeff=1.5, quant = 0.005, include_boundaries=False):
        """
        This method takes in a DataFrame or Series and returns True for each element that is an outlier according to
        a mehod specified.

        Parameters
        ----------
        data : DataFrame or Series of timeseries data
        method : str, default "std
            "std" : bounds are  +- coeff * std_dev around the mean of each time series (constant in time)
            "iqr" : bounds are  +- coeff * interquartile range (constant in time)
            "qtl": bounds are outside of a specific quantile (const in time)
            "rstd" : bounds are  +- coeff * std_dev around the mean of each time series in a window of n timestamps (time-varying)
            "rmedian" : same as above but around the meadian
        coeff : int
            coeff to be multiplied to std dev 
        window : int, default 10
            window size (n rows) for rolling variants
        qcoeff : float or int, default 1.5
            coeff to be multiplied to iqr
        quant : float < 1, default 0.005
            quantile
        include_boundaries : bool, default False
            wether to use ">" or ">=" with bounds

        Returns
        -------
        DataFrames or Series with True where the condition element is an outlier, according to the method
        
        """
        lowBound, highBound = self._calc_outliers_bounds(data=data, method=method, coeff=coeff, window=window,
                                                        qcoeff=qcoeff, quant=quant)
        
        #TODO: there may be method to identify outliers that do not generate low and high bounds, but select outliers directly 

        return self._func_outOfBound(data, lowBound, highBound, include_boundaries)
    
        
    
    
########################################################################################################

    def find_missing(self, data, return_bool=False, how="any"):
    
        return self._find(data, self._func_missing, return_bool=return_bool, how=how)


    def display_missing(self, data, how="any"):

        return self._display(data, self._func_missing, how=how, mask_others=False)


    def remove_missing(self, data, return_bool=True, how="any"):

        return self._remove(data, self._func_missing, return_bool=return_bool, how=how, mask_others=False)

    
    def count_missing(self, data, output="number"):

        return self._count(data, self._func_missing, output=output)
        
########################################################################################################

    def find_flatlines(self, data, return_bool=False, how="any"):
    
        return self._find(data, self._func_flatlines, return_bool=return_bool, how=how)


    def display_flatlines(self, data, how="any"):

        return self._display(data, self._func_flatlines, how=how, mask_others=True)


    def remove_flatlines(self, data, return_bool=True, how="any"):

        return self._remove(data, self._func_flatlines, return_bool=return_bool, how=how, mask_others=True)
        

    def count_flatlines(self, data, output="number"):

        return self._count(data, self._func_flatlines, output=output)
    

########################################################################################################

    def find_if(self, data, operator, val, return_bool=False, how="any"):
    
        return self._find(data, self._func_if, return_bool=return_bool, how=how, operator=operator, val=val)


    def display_if(self, data, operator, val, how="any"):

        return self._display(data, self._func_if, how=how, mask_others=True, operator=operator, val=val)


    def remove_if(self, data, operator, val, return_bool=True, how="any"):

        return self._remove(data, self._func_if, return_bool=return_bool, how=how, mask_others=True, operator=operator, val=val)

    
    def count_if(self, data, operator, val, output="number"):

        return self._count(data, self._func_if, output=output, operator=operator, val=val)
    

########################################################################################################

    def find_outOfBound(self, data, lowBound, highBound, return_bool=False, how="any"):
    
        return self._find(data, self._func_outOfBound, return_bool=return_bool, how=how, lowBound=lowBound, highBound=highBound)


    def display_outOfBound(self, data, lowBound, highBound, how="any"):

        return self._display(data, self._func_outOfBound, how=how, mask_others=True, lowBound=lowBound, highBound=highBound)

    
    def remove_outOfBound(self, data, lowBound, highBound,  return_bool=True, how="any"):

        return self._remove(data, self._func_outOfBound, return_bool=return_bool, how=how, mask_others=True, lowBound=lowBound, highBound=highBound)

    
    def count_outOfBound(self, data, lowBound, highBound,  output="number"):

        return self._count(data, self._func_outOfBound, output=output, lowBound=lowBound, highBound=highBound)
        
########################################################################################################

    def find_outliers(self, data, method="std", coeff=3, window=10, return_bool=False, how="any"):
    
        return self._find(data, self._func_outliers, return_bool=return_bool, how=how, method=method, coeff=coeff, window=window)


    def display_outliers(self, data, method="std", coeff=3, window=10, how="any"):

        return self._display(data, self._func_outliers, how=how, mask_others=True, method=method, coeff=coeff, window=window)

    
    def remove_outliers(self, data, method="std", coeff=3, window=10, return_bool=True, how="any"):

        return self._remove(data, self._func_outliers, return_bool=return_bool, how=how, mask_others=True, method=method, coeff=coeff, window=window)


    def count_outliers(self, data, method="std", coeff=3, window=10, output="number"):

        return self._count(data, self._func_outliers, output=output, method=method, coeff=coeff, window=window)
    

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