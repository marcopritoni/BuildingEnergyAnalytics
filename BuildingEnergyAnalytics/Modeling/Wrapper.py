'''
This script is a wrapper class around all the different modules - importing, cleaning, preprocessing and modeling the data.

TODO
1. Add parameter that exlcudes certain time_periods.
2. Dump metadata of optimal model. Also, save dataframe pulled from database.
3. Save matplotlib graphs.
4. Automate high_bound of data.
5. Add option to standardize/normalize data before fitting to model.
6. Add TimeSeriesSplit, ANN, SVM.
7. Add percent error, NMBE in Model_Data.py/display_metrics().
8. Dump data into json file.

Cleanup
1. Delete unusued variables.
2. Run pylint on all files.
3. Documentation.

Note
1. df.loc[(slice(None, None, None)), ...] is equivalent to "df.loc[:,...]"
2. df.resample(freq='h').mean() drops all non-float/non-int columns
3. os._exit(1) exits the program without calling cleanup handlers.

Last modified: August 8 2018
@author Pranav Gupta <phgupta@ucdavis.edu>
'''

import datetime
import numpy as np
import pandas as pd
from Import_Data import *
from Clean_Data import *
from Preprocess_Data import *
from Model_Data import *

class Wrapper:

    def __init__(self, result_filename=None):

        self.imported_data = pd.DataFrame()
        self.cleaned_data = pd.DataFrame()
        self.preprocessed_data = pd.DataFrame()
        self.X = pd.DataFrame()
        self.y = pd.DataFrame()
        self.metrics = {}

        if not result_filename:
            result_filename = 'results.txt'

        self.f = open(result_filename, 'a')
        self.f.write('\n\nTime: {}\n\n'.format(datetime.datetime.now())) # Local timezone

    def __del__(self):
        ''' Destructor '''
        self.f.close()


    def import_data(self, file_name=None, folder_name=None, head_row=0, index_col=0, 
                    convert_col=True, concat_files=False):

        # print("Importing data...")
        self.f.write('Importing data...\n')
        
        import_data_obj = Import_Data(self.f)
        import_data_obj.import_csv(file_name=file_name, folder_name=folder_name, 
            head_row=head_row, index_col=index_col, convert_col=convert_col, concat_files=concat_files)
        self.imported_data = import_data_obj.data
        
        # print('{:*^50}\n'.format('Successfully imported data!'))
        self.f.write('{:*^50}\n\n'.format('Finished importing data!'))
        return self.imported_data


    def clean_data(self, data, rename_col=None, resample=True, freq='h', interpolate=True, limit=1, 
                    remove_na=True, remove_na_how='any', remove_outliers=True, sd_val=3, 
                    remove_out_of_bounds=True, low_bound=0, high_bound=float('inf')):

        # print("Cleaning data...")
        self.f.write('Cleaning data...\n')
        
        clean_data_obj = Clean_Data(data, self.f)
        clean_data_obj.clean_data(resample=resample, freq=freq, interpolate=interpolate, 
                                limit=limit, remove_na=remove_na, remove_na_how=remove_na_how, 
                                remove_outliers=remove_outliers, sd_val=sd_val, 
                                remove_out_of_bounds=remove_out_of_bounds, 
                                low_bound=low_bound, high_bound=high_bound)
        if rename_col:
            clean_data_obj.rename_columns(rename_col)
        
        self.cleaned_data = clean_data_obj.cleaned_data
        
        # print('{:*^50}\n'.format('Successfully cleaned data!'))
        self.f.write('{:*^50}\n\n'.format('Finished cleaning data!'))
        return self.cleaned_data


    def preprocess_data(self, data, input_col_degree=None, degree=None, 
                        YEAR=False, MONTH=False, WEEK=False, TOD=False, DOW=False, DOY=False, 
                        hdh_cpoint=65, cdh_cpoint=65, hdh_cdh_calc_col='OAT', var_to_expand=None):
        
        # print("Preprocessing data...")
        self.f.write('Preprocessing data...\n')
        
        preprocess_data_obj = Preprocess_Data(data, self.f)
        preprocess_data_obj.add_degree_days(col=hdh_cdh_calc_col, hdh_cpoint=hdh_cpoint, cdh_cpoint=cdh_cpoint)
        preprocess_data_obj.add_col_features(input_col=input_col_degree, degree=degree)
        preprocess_data_obj.add_time_features(YEAR=YEAR, MONTH=MONTH, WEEK=WEEK, 
                                                TOD=TOD, DOW=DOW, DOY=DOY)
        preprocess_data_obj.create_dummies(var_to_expand=var_to_expand)
        self.preprocessed_data = preprocess_data_obj.preprocessed_data
        
        # print('{:*^50}\n'.format('Successfully preprocessed data!'))
        self.f.write('{:*^50}\n\n'.format('Finished preprocessing data!'))
        return self.preprocessed_data


    def model(self, data, output_col, alphas=np.logspace(-4,1,30),
            time_period=None, input_col=None, plot=True, figsize=None, custom_model_func=None):

        # print("Splitting data...")
        self.f.write('Splitting data...\n')
        
        model_data_obj = Model_Data(data, self.f, time_period, output_col, alphas, input_col)
        model_data_obj.split_data()
        self.X = model_data_obj.baseline_period_in
        self.y = model_data_obj.baseline_period_out
        # print('{:*^50}\n'.format('Successfully split data!'))
        self.f.write('{:*^50}\n\n'.format('Finished splitting data!'))

        # print("Running different models...")
        self.f.write('Model selection...\n')
        best_model, best_model_name = model_data_obj.run_models()
        # print('{:*^50}\n'.format('Successfully ran all models!'))
        self.f.write('{:*^50}\n\n'.format('Finished running all models!'))

        if custom_model_func:
            # print('Running custom model...')
            self.f.write('Running custom model function...\n')
            model_data_obj.custom_model(custom_model_func)
            # print('{:*^50}\n'.format('Successfully ran custom model function!'))
            self.f.write('{:*^50}\n\n'.format('Finished running custom model function!'))

        # print("Fitting data to Best Model: ", best_model_name)
        self.f.write('Choosing best model: {}\n'.format(best_model_name))
        model_data_obj.best_model_fit(best_model)
        # print('{:*^50}\n'.format('Successfully ran best model!'))
        self.f.write('{:*^50}\n\n'.format('Successfully fit data to best model!'))

        # print("Displaying metrics...")
        self.f.write('Displaying metrics...\n')
        self.metrics = model_data_obj.display_metrics()
        # print('{:*^50}\n'.format('Successfully displayed metrics!'))
        self.f.write('{:*^50}\n\n'.format('Finished displaying metrics!'))

        if plot:
            self.f.write('Displaying plots...\n')
            model_data_obj.display_plots(figsize)
        
        # print('{:*^50}\n'.format('Successfully modeled data!'))
        self.f.write('{:*^50}\n\n'.format('Finished modeling data!'))

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