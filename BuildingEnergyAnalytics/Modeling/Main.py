'''
This script is a wrapper class around all the different modules - importing, cleaning, preprocessing and modeling the data.

TODO
1. Use TimeSeriesSplit - https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/
2. Ensure Import_Data's functions match that of Influx & Skyspark's class. 
3. Standardize/Normalize data before fitting in model?
4. Make time_periods flexible (i.e. don't restrict to 6 values only)
5. Add all metrics in calc_scores - NMBE, CV_RMSE, RMSE...

Last modified: July 16 2018
@author Pranav Gupta <phgupta@ucdavis.edu>
'''

import pandas as pd
from Import_Data import *
from Clean_Data import *
from Preprocess_Data import *
from Model_Data import *
from sklearn.linear_model import LinearRegression

class Main:

	def __init__(self,
		file_name=None, folder_name=None, head_row=0, index_col=0, convert_col=True, concat_files=False,
		rename_col=None, resample=True, freq='h', interpolate=True, limit=1, remove_na=True, remove_na_how='any', remove_outliers=True, sd_val=3, remove_out_of_bounds=True, low_bound=0, high_bound=9998,
		input_col_degree=None, degree=None, output_col=None, YEAR=False, MONTH=False, WEEK=True, TOD=True, DOW=False, DOY=False, hdh_cpoint=65, cdh_cpoint=65, hdh_cdh_calc_col='OAT', var_to_expand=['TOD','DOW', 'WEEK'],
		time_period=None, input_col=None, plot=True):
		'''
			Constructor!
			First line contains parameters for importing data
			Second line contains parameters for cleaning 
			Third line contains parameters for preprocessing the data
			Fourth line contains parameters for fitting the model
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
			
		self.time_period = time_period
		self.input_col = input_col
		self.plot = plot

		self.imported_data = pd.DataFrame()
		self.cleaned_data = pd.DataFrame()
		self.preprocessed_data = pd.DataFrame()
		self.X = pd.DataFrame()
		self.y = pd.DataFrame()


	def import_data(self):

		print("Importing data...")
		
		import_data_obj = Import_Data()
		import_data_obj.import_csv(file_name=self.file_name, folder_name=self.folder_name, 
			head_row=self.head_row, index_col=self.index_col, convert_col=self.convert_col, concat_files=self.concat_files)
		
		self.imported_data = import_data_obj.data
		print('{:*^50}\n'.format('Successfully imported data!'))

		return self.imported_data


	def clean_data(self, data):

		print("Cleaning data...")
		
		clean_data_obj = Clean_Data(data)
		
		clean_data_obj.clean_data(resample=self.resample, freq=self.freq, interpolate=self.interpolate, 
								limit=self.limit, remove_na=self.remove_na, remove_na_how=self.remove_na_how, 
								remove_outliers=self.remove_outliers, sd_val=self.sd_val, 
								remove_out_of_bounds=self.remove_out_of_bounds, 
								low_bound=self.low_bound, high_bound=self.high_bound)
		if self.rename_col:
			clean_data_obj.rename_columns(self.rename_col)
		
		self.cleaned_data = clean_data_obj.cleaned_data
		print('{:*^50}\n'.format('Successfully cleaned data!'))

		return self.cleaned_data


	def preprocess_data(self, data):

		print("Preprocessing data...")
		
		preprocess_data_obj = Preprocess_Data(data)
		preprocess_data_obj.add_degree_days(col=self.hdh_cdh_calc_col, hdh_cpoint=self.hdh_cpoint, cdh_cpoint=self.cdh_cpoint)
		preprocess_data_obj.add_col_features(input_col=self.input_col_degree, degree=self.degree)
		preprocess_data_obj.add_time_features(YEAR=self.YEAR, MONTH=self.MONTH, WEEK=self.WEEK, 
												TOD=self.TOD, DOW=self.DOW, DOY=self.DOY)
		preprocess_data_obj.create_dummies(var_to_expand=self.var_to_expand)
		
		self.preprocessed_data = preprocess_data_obj.preprocessed_data
		print('{:*^50}\n'.format('Successfully preprocessed data!'))

		return self.preprocessed_data


	def split_model_data(self, data):

		print("Splitting data...")
		
		model_data_obj = Model_Data(data, self.time_period, self.output_col, input_col=self.input_col)
		model_data_obj.split_data()
		
		self.X = model_data_obj.baseline_period_in
		self.y = model_data_obj.baseline_period_out
		print('{:*^50}\n'.format('Successfully split data!'))

		print("Running different models...")
		best_model, model_name = model_data_obj.run_models()

		print("Fitting data to ", model_name)
		model_data_obj.best_model_fit(best_model)

		print("Displaying metrics...")
		model_data_obj.display_metrics()

		# if self.plot:
		# 	model_data_obj.display_plots()

		print('{:*^50}\n'.format('Successfully modeled data!'))


if __name__ == '__main__':
	
	'''
	Absolutely necessary to fill,
		1. file_name/folder_name (data source)
		2. output_col (column to predict)
		3. model (Linear Regression, Lasso, Ridge...)
		4. time_period (for splitting the data)
	'''
	
	################# IMPORT DATA FROM CSV FILES #################
	# main_obj = Main(
	# 		folder_name='../../../../../Desktop/LBNL/Data/', head_row=[5,5,0], 
	# 		rename_col=['OAT', 'RelHum_Avg', 'CHW_Elec', 'Elec', 'Gas', 'HW_Heat'],
	# 		output_col='HW_Heat', MONTH=True, var_to_expand=['TOD', 'WEEK', 'MONTH'],
	# 		time_period=["2014-01","2014-12", "2015-01","2015-12", "2016-01","2016-12"]
	# 	)

	# imported_data = main_obj.import_data()
	# cleaned_data = main_obj.clean_data(imported_data)
	# preprocessed_data = main_obj.preprocess_data(cleaned_data)
	# main_obj.split_model_data(preprocessed_data)

	
	################# IMPORT DATA FROM INFLUXDB #################
	# import configparser
	# from influxdb import DataFrameClient

	# config = configparser.ConfigParser()
	# config.read('./config.ini')
	# host = config['CREDENTIALS']['host']
	# port = config['CREDENTIALS']['port']
	# database = config['CREDENTIALS']['database']
	# username = config['CREDENTIALS']['username']
	# password = config['CREDENTIALS']['password']

	# meter_names = ['\'r:p:lbnl:r:2193afbc-27e92439 BACnet JCI 15.0 Power (BTU/h)\'',
    #          '\'r:p:lbnl:r:21954442-d84f76af LBNL Weather Station Dry-Bulb Temperature Sensor (Main, Two-Meter) (°F)\'']
	# column_names = ['Power', 'OAT']

	# client = DataFrameClient(host=host, port=port, database=database, 
	# 						username=username, password=password, ssl=True, verify_ssl=True)

	# df = pd.DataFrame()
	# for i, meter_name in enumerate(meter_names):
	#     query = 'SELECT Value FROM pyTestDB.autogen.Skyspark_Analysis WHERE time > \'2017-11-05T06:45:00.000Z\' AND \"Meter Name\"=' + meter_name
	#     result = client.query(query)
	#     result = result['Skyspark_Analysis']
	#     result.columns = [column_names[i]]
	#     df = df.join(result, how='outer')

	# df = df.resample('h').mean()

	# main_obj = Main( 
	# 		remove_out_of_bounds=False,
	# 		input_col=['OAT'], degree=[2], output_col='Power', MONTH=True, var_to_expand=['TOD', 'WEEK', 'MONTH'],
	# 		model=LinearRegression(), time_period=["2017-11", "2018-06", None, None, None, None]
	# 	)

	# cleaned_data = main_obj.clean_data(df)
	# preprocessed_data = main_obj.preprocess_data(cleaned_data)
	# main_obj.split_model_data(preprocessed_data)
