'''
This script is a wrapper class around all the different modules - importing, cleaning, preprocessing and modeling the data.

To Do
1. Model
    1. Add TimeSeriesSplit, ANN, SVM, Randomforest.
    2. Add option to standardize/normalize data before fitting to model.
    3. Add percent error, NMBE in Model_Data.py/display_metrics().
    4. Add cv parameter in model_data()
2. Wrapper
    1. Dump data into json file.
    2. Add all function parameters in json.
    3. Give user the option to run specific models.
    4. Write function to read json file which runs run_all().
    5. Run iterations on resampling frequency, adding time features (TOD, DOW, DOY...)
    6. Add Pearson's correlation coefficient.
3. All
    1. Change SystemError to specific errors.

Cleanup
1. Delete unusued variables.
2. Run pylint on all files.
3. Documentation.
4. Structure code to publish to PyPI.

Notes
1. df.loc[(slice(None, None, None)), ...] is equivalent to "df.loc[:,...]"
2. df.resample(freq='h').mean() drops all non-float/non-int columns
3. os._exit(1) exits the program without calling cleanup handlers.

Last modified: August 25 2018
@author Pranav Gupta <phgupta@ucdavis.edu>
'''

import os
import json
import datetime
import numpy as np
import pandas as pd
from Import_Data import *
from Clean_Data import *
from Preprocess_Data import *
from Model_Data import *


class Wrapper:

    def __init__(self, results_folder_name='results'):
        ''' Constructor '''

        self.imported_data = pd.DataFrame()
        self.cleaned_data = pd.DataFrame()
        self.preprocessed_data = pd.DataFrame()
        self.results_folder_name = results_folder_name
        self.result = {}

        self.X = pd.DataFrame()
        self.y = pd.DataFrame()
        self.metrics = {}

        # UTC Time
        self.result['Time (UTC)'] = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

        # Create results folder if it doesn't exist
        if not os.path.isdir(self.results_folder_name):
            os.makedirs(self.results_folder_name)

    def write_json(self):
        ''' Dump data into json '''

        # If no errors, set error = ''
        # if not 'Error' in self.result:
        #     self.result['Error'] = ''

        # Dump json data to results.json file
        with open(self.results_folder_name+'/results.json', 'a') as f:
            json.dump(self.result, f)


    def import_data(self, file_name='*', folder_name='.', head_row=0, index_col=0, 
                    convert_col=True, concat_files=False, save_file=True):
        ''' 
            Import data from CSV, Influx, MongoDB...
            Currently this function supports CSV files only.
        '''
        
        # Create instance and import the data
        import_data_obj = Import_Data()
        import_data_obj.import_csv(file_name=file_name, folder_name=folder_name, 
            head_row=head_row, index_col=index_col, convert_col=convert_col, concat_files=concat_files)
        
        # Store imported data in wrapper class
        self.imported_data = import_data_obj.data

        # Logging
        self.result['Import'] = {
            'Source': 'CSV', # import_data() supports only csv files currently
            'File Name': file_name,
            'Folder Name': folder_name,
            'Head Row': head_row,
            'Index Col': index_col,
            'Convert Col': convert_col,
            'Concat Files': concat_files
        }
        
        if save_file:
            self.imported_data.to_csv(self.results_folder_name + '/imported_data.csv')
            self.result['Import']['Saved File'] = self.results_folder_name + '/imported_data.csv'
        else:
            self.result['Import']['Saved File'] = ''

        return self.imported_data


    def clean_data(self, data, rename_col=None, drop_col=None,
                    resample=True, freq='h',
                    interpolate=True, limit=1, method='linear',
                    remove_na=True, remove_na_how='any',
                    remove_outliers=True, sd_val=3,
                    remove_out_of_bounds=True, low_bound=0, high_bound=float('inf'),
                    save_file=True):
        '''
            Cleans data by resampling, interpolating, removing outliers, NA's...

        '''

        # Check to ensure data is a pandas series or dataframe
        if not isinstance(data, pd.DataFrame) and not isinstance(data, pd.Series):
            raise SystemError('data has to be a pandas dataframe or series.')
        
        # Create instance and clean the data
        clean_data_obj = Clean_Data(data)
        clean_data_obj.clean_data(resample=resample, freq=freq, interpolate=interpolate, 
                                limit=limit, remove_na=remove_na, remove_na_how=remove_na_how, 
                                remove_outliers=remove_outliers, sd_val=sd_val, 
                                remove_out_of_bounds=remove_out_of_bounds, 
                                low_bound=low_bound, high_bound=high_bound)
        if rename_col: # Rename columns of dataframe
            clean_data_obj.rename_columns(rename_col)
        if drop_col: # Drop columns of dataframe
            clean_data_obj.drop_columns(drop_col)
        
        # Store cleaned data in wrapper class
        self.cleaned_data = clean_data_obj.cleaned_data

        # Logging
        self.result['Clean'] = {
            'Rename Col': rename_col,
            'Drop Col': drop_col,
            'Resample': resample,
            'Frequency': freq,
            'Interpolate': interpolate,
            'Limit': limit,
            'Method': method,
            'Remove NA': remove_na,
            'Remove NA How': remove_na_how,
            'Remove Outliers': remove_outliers,
            'SD Val': sd_val,
            'Remove Out of Bounds': remove_out_of_bounds,
            'Low Bound': low_bound,
            'High Bound': high_bound
        }

        if self.imported_data.empty:
            self.result['Clean']['Source'] = '' # User provided their own dataframe, i.e. they did not use import_data()
        else:
            self.result['Clean']['Source'] = self.results_folder_name + '/imported_data.csv'

        if save_file:
            self.cleaned_data.to_csv(self.results_folder_name + '/cleaned_data.csv')
            self.result['Clean']['Saved File'] = self.results_folder_name + '/cleaned_data.csv'
        else:
            self.result['Clean']['Saved File'] = ''

        return self.cleaned_data


    def preprocess_data(self, data,
                        hdh_cpoint=65, cdh_cpoint=65, col_hdh_cdh='OAT', 
                        col_degree=None, degree=None, 
                        year=False, month=False, week=False, tod=False, dow=False, doy=False,  
                        var_to_expand=None, 
                        save_file=True):
        '''
            Preprocesses data by adding time features and exponentiating certain columns

        '''

        # Check to ensure data is a pandas series or dataframe
        if not isinstance(data, pd.DataFrame) and not isinstance(data, pd.Series):
            raise SystemError('data has to be a pandas dataframe or series.')
        
        # Create instance
        preprocess_data_obj = Preprocess_Data(data)
        preprocess_data_obj.add_degree_days(col=col_hdh_cdh, hdh_cpoint=hdh_cpoint, cdh_cpoint=cdh_cpoint)
        preprocess_data_obj.add_col_features(col=col_degree, degree=degree)
        preprocess_data_obj.add_time_features(year=year, month=month, week=week, tod=tod, dow=dow, doy=doy)
        preprocess_data_obj.create_dummies(var_to_expand=var_to_expand)
        
        # Store preprocessed data in wrapper class
        self.preprocessed_data = preprocess_data_obj.preprocessed_data

        # Logging
        self.result['Preprocess'] = {
            'HDH CPoint': hdh_cpoint,
            'CDH CPoint': cdh_cpoint,
            'HDH CDH Calc Col': col_hdh_cdh,
            'Col Degree': col_degree,
            'Degree': degree,
            'Year': year,
            'Month': month,
            'Week': week,
            'Time of Day': tod,
            'Day of Week': dow,
            'Day of Year': doy,
            'Variables to Expand': var_to_expand
        }

        if self.cleaned_data.empty:
            self.result['Preprocess']['Source'] = '' # User provided their own dataframe, i.e. they did not use cleaned_data()
        else:
            self.result['Preprocess']['Source'] = self.results_folder_name + '/cleaned_data.csv'

        if save_file:
            self.preprocessed_data.to_csv(self.results_folder_name + '/preprocessed_data.csv')
            self.result['Preprocess']['Saved File'] = self.results_folder_name + '/preprocessed_data.csv'
        else:
            self.result['Preprocess']['Saved File'] = ''

        return self.preprocessed_data


    def model(self, data, output_col, alphas=np.logspace(-4,1,30), time_period=[None,None], exclude_time_period=[None,None],
            input_col=None, plot=True, figsize=None, custom_model_func=None):

        # if not data or not a DataFrame/Series:
        #     raise error
        
        model_data_obj = Model_Data(data, time_period, exclude_time_period, output_col, alphas, input_col)
        model_data_obj.split_data()
        
        self.X = model_data_obj.baseline_period_in
        self.y = model_data_obj.baseline_period_out
        
        best_model, best_model_name = model_data_obj.run_models()

        if custom_model_func:
            model_data_obj.custom_model(custom_model_func)

        model_data_obj.best_model_fit(best_model)

        self.metrics = model_data_obj.display_metrics()

        if plot:
            fig1, fig2 = model_data_obj.display_plots(figsize)
            fig1.savefig(self.results_folder_name+'/acc_alpha.png')
            fig2.savefig(self.results_folder_name+'/modeled_data.png')
        
        return self.metrics


if __name__ == '__main__':
        
    ################ IMPORT DATA FROM CSV FILES #################

    def func(X, y):
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_score
        model = LinearRegression()
        model.fit(X, y)
        return model.predict(X)

    wrapper_obj = Main()

    imported_data = wrapper_obj.import_data(folder_name='../../../../../Desktop/LBNL/Data/', head_row=[5,5,0])
    cleaned_data = wrapper_obj.clean_data(imported_data, high_bound=9998,
                                    rename_col=['OAT', 'RelHum_Avg', 'CHW_Elec', 'Elec', 'Gas', 'HW_Heat'])
    preprocessed_data = wrapper_obj.preprocess_data(cleaned_data, WEEK=True, TOD=True, var_to_expand=['TOD','WEEK'])
    wrapper_obj.model(preprocessed_data, output_col='HW_Heat', alphas=np.logspace(-4,1,5), figsize=(18,5),
                    time_period=["2014-01","2014-12", "2015-01","2015-12", "2016-01","2016-12"],
                    custom_model_func=func)