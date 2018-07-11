'''
This script is a wrapper class around all the different modules - importing, cleaning, preprocessing and modeling the data.

TODO
1. Make all the module functions independent (add a "data" parameter)
2. Use TimeSeriesSplit - https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/
3. Before fitting model, print input_col and output_col
4. Drop last column to remove multi-collinearity

Last modified: July 11 2018
@author Pranav Gupta <phgupta@ucdavis.edu>
'''

import pandas as pd
from Import_Data import *
from Clean_Data import *
from Preprocess_Data import *
from Model_Data import *

class Main:

	def __init__(self,
		file_name=None, folder_name=None, head_row=0, index_col=0, convert_col=True, concat_files=False,
		rename_col=None, resample=True, freq='h', interpolate=True, limit=1, remove_na=True, remove_na_how='any', remove_outliers=True, sd_val=3, remove_out_of_bounds=True, low_bound=0, high_bound=9998,
		input_col_degree=None, degree=None, output_col=None, YEAR=False, MONTH=False, WEEK=True, TOD=True, DOW=True, DOY=False, hdh_cpoint=65, cdh_cpoint=65, hdh_cdh_calc_col='OAT', var_to_expand=['TOD','DOW', 'WEEK'],
		model=None, time_period=None, input_col=None, plot=True):
		'''
			Constructor!
			First line contains parameters for importing data
			Second line contains parameters for cleaning 
			Third line contains parameters for preprocessing the data
			Third line contains parameters for fitting the model

			Necessary to fill,
			1. file_name/folder_name (data source)
			2. output_col (col to predict)
			3. model (Linear Regression, Lasso, Ridge...)
			4. time_period (for splitting the data)
		'''
		self.file_name = file_name
		self.folder_name = folder_name 
		self.head_row = head_row
		self.index_col = index_col
		self.convert_col = convert_col 
		self.concat_files = concat_files

		self.rename_col = rename_col
		self.resample = resample
		self.freq = freq 
		self.interpolate = interpolate
		self.limit = limit
		self.remove_na = remove_na
		self.remove_na_how =  remove_na_how 
		self.remove_outliers = remove_outliers
		self.sd_val = sd_val
		self.remove_out_of_bounds = remove_out_of_bounds
		self.low_bound = low_bound 
		self.high_bound = high_bound

		self.input_col_degree = input_col_degree
		self.degree = degree 
		self.output_col = output_col
		self.YEAR = YEAR
		self.MONTH = MONTH
		self.WEEK = WEEK
		self.TOD = TOD 
		self.DOW = DOW
		self.DOY = DOY
		self.hdh_cpoint = hdh_cpoint 
		self.cdh_cpoint = cdh_cpoint
		self.hdh_cdh_calc_col = hdh_cdh_calc_col 
		self.var_to_expand = var_to_expand
			
		self.model = model
		self.time_period = time_period
		self.input_col = input_col
		self.plot = plot

		self.imported_data = pd.DataFrame()
		self.cleaned_data = pd.DataFrame()
		self.preprocessed_data = pd.DataFrame()
		self.X = pd.DataFrame()
		self.y = pd.DataFrame()


	def run(self):
		''' Run an entire cycle of import, clean, preprocess and model data '''

		print("Importing data...")
		import_data_obj = Import_Data()
		import_data_obj.import_csv(file_name=self.file_name, folder_name=self.folder_name, 
			head_row=self.head_row, index_col=self.index_col, convert_col=self.convert_col, concat_files=self.concat_files)
		self.imported_data = import_data_obj.data
		print("*****Successfully imported data!*****")

		print("Cleaning data...")
		clean_data_obj = Clean_Data(self.imported_data)
		clean_data_obj.clean_data()
		if self.rename_col:
			clean_data_obj.rename_columns(self.rename_col)
		self.cleaned_data = clean_data_obj.cleaned_data
		print("*****Successfully cleaned data!*****")

		print("Preprocessing data...")
		preprocess_data_obj = Preprocess_Data(self.cleaned_data)
		preprocess_data_obj.add_degree_days(col=self.hdh_cdh_calc_col, hdh_cpoint=self.hdh_cpoint, cdh_cpoint=self.cdh_cpoint)
		preprocess_data_obj.add_col_features(input_col=self.input_col_degree, degree=self.degree)
		preprocess_data_obj.add_time_features(YEAR=self.YEAR, MONTH=self.MONTH, WEEK=self.WEEK, TOD=self.TOD, DOW=self.DOW, DOY=self.DOY)
		preprocess_data_obj.create_dummies(var_to_expand=self.var_to_expand)
		self.preprocessed_data = preprocess_data_obj.preprocessed_data
		print("*****Successfully preprocessed data!*****")

		print("Splitting data...")
		model_data_obj = Model_Data(self.preprocessed_data)
		model_data_obj.split_data(time_period=self.time_period, input_col=self.input_col, output_col=self.output_col)
		self.X = model_data_obj.baseline_period_in
		self.y = model_data_obj.baseline_period_out
		print("*****Successfully split data!*****")


if __name__ == '__main__':
	
	main_obj = Main(
			folder_name='../Data/', head_row=[5,5,0], 
			rename_col=['OAT', 'RelHum_Avg', 'CHW_Elec', 'Elec', 'Gas', 'HW_Heat'],
			output_col='HW_Heat',
			time_period=["2014-01","2014-12", "2015-01","2015-12", "2016-01","2016-12"]
		)

	main_obj.run()