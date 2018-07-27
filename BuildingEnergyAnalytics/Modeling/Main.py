'''
This script is a wrapper class around all the different modules - importing, cleaning, preprocessing and modeling the data.

TODO
1. Run all the different models (cross validating with different alphas) and log them to a file. 
2. Run the model with the best score with alphas in the range (alpha-x, alpha+x)
3. Add TimeSeriesSplit, ANN, SVM.
4. Standardize/Normalize data before fitting to model.
5. Add all metrics in calc_scores - NMBE, CV_RMSE, RMSE...
6. Make time_periods flexible (i.e. don't restrict to 6 values only)
7. Figure out which variables to save.

Bugs
1. When getting data from influxdb, 
	1. df = df.resample('d').mean() is wrong!
	2. df.loc[(slice(None, None, None)), ...] does not give an error.

Last modified: July 27 2018
@author Pranav Gupta <phgupta@ucdavis.edu>
'''

import pandas as pd
from Import_Data import *
from Clean_Data import *
from Preprocess_Data import *
from Model_Data import *
from sklearn.linear_model import LinearRegression

class Main:

	def __init__(self):

		self.imported_data = pd.DataFrame()
		self.cleaned_data = pd.DataFrame()
		self.preprocessed_data = pd.DataFrame()
		self.X = pd.DataFrame()
		self.y = pd.DataFrame()


	def import_data(self, file_name=None, folder_name=None, head_row=0, index_col=0, 
					convert_col=True, concat_files=False):

		print("Importing data...")
		
		import_data_obj = Import_Data()
		import_data_obj.import_csv(file_name=file_name, folder_name=folder_name, 
			head_row=head_row, index_col=index_col, convert_col=convert_col, concat_files=concat_files)
		self.imported_data = import_data_obj.data
		
		print('{:*^50}\n'.format('Successfully imported data!'))
		return self.imported_data


	def clean_data(self, data, rename_col=None, resample=True, freq='h', interpolate=True, limit=1, 
					remove_na=True, remove_na_how='any', remove_outliers=True, sd_val=3, 
					remove_out_of_bounds=True, low_bound=0, high_bound=float('inf')):

		print("Cleaning data...")
		
		clean_data_obj = Clean_Data(data)
		clean_data_obj.clean_data(resample=resample, freq=freq, interpolate=interpolate, 
								limit=limit, remove_na=remove_na, remove_na_how=remove_na_how, 
								remove_outliers=remove_outliers, sd_val=sd_val, 
								remove_out_of_bounds=remove_out_of_bounds, 
								low_bound=low_bound, high_bound=high_bound)
		if rename_col:
			clean_data_obj.rename_columns(rename_col)
		self.cleaned_data = clean_data_obj.cleaned_data
		
		print('{:*^50}\n'.format('Successfully cleaned data!'))
		return self.cleaned_data


	def preprocess_data(self, data, input_col_degree=None, degree=None, 
						YEAR=False, MONTH=False, WEEK=False, TOD=False, DOW=False, DOY=False, 
						hdh_cpoint=65, cdh_cpoint=65, hdh_cdh_calc_col='OAT', var_to_expand=None):
		
		print("Preprocessing data...")
		
		preprocess_data_obj = Preprocess_Data(data)
		preprocess_data_obj.add_degree_days(col=hdh_cdh_calc_col, hdh_cpoint=hdh_cpoint, cdh_cpoint=cdh_cpoint)
		preprocess_data_obj.add_col_features(input_col=input_col_degree, degree=degree)
		preprocess_data_obj.add_time_features(YEAR=YEAR, MONTH=MONTH, WEEK=WEEK, 
												TOD=TOD, DOW=DOW, DOY=DOY)
		preprocess_data_obj.create_dummies(var_to_expand=var_to_expand)
		self.preprocessed_data = preprocess_data_obj.preprocessed_data
		
		print('{:*^50}\n'.format('Successfully preprocessed data!'))
		return self.preprocessed_data


	def model(self, data, output_col=None, time_period=None, input_col=None, plot=True):

		print("Splitting data...")
		
		model_data_obj = Model_Data(data, time_period, output_col, input_col=input_col)
		model_data_obj.split_data()
		self.X = model_data_obj.baseline_period_in
		self.y = model_data_obj.baseline_period_out
		print('{:*^50}\n'.format('Successfully split data!'))

		print("Running different models...")
		best_model, best_model_name = model_data_obj.run_models()
		print('{:*^50}\n'.format('Successfully ran all models!'))

		print("Fitting data to Best Model: ", best_model_name)
		model_data_obj.best_model_fit(best_model)
		print('{:*^50}\n'.format('Successfully ran best model!'))

		print("Displaying metrics...")
		model_data_obj.display_metrics()
		print('{:*^50}\n'.format('Successfully displayed metricsl!'))

		if plot:
			model_data_obj.display_plots()
		
		print('{:*^50}\n'.format('Successfully modeled data!'))


if __name__ == '__main__':
		
	################ IMPORT DATA FROM CSV FILES #################
	main_obj = Main()

	imported_data = main_obj.import_data(folder_name='../../../../../Desktop/LBNL/Data/', head_row=[5,5,0])
	cleaned_data = main_obj.clean_data(imported_data, high_bound=9998,
									rename_col=['OAT', 'RelHum_Avg', 'CHW_Elec', 'Elec', 'Gas', 'HW_Heat'])
	preprocessed_data = main_obj.preprocess_data(cleaned_data, WEEK=True, TOD=True, var_to_expand=['TOD','WEEK'])
	main_obj.model(preprocessed_data, output_col='HW_Heat', 
					time_period=["2014-01","2014-12", "2015-01","2015-12", "2016-01","2016-12"])

	# ################# IMPORT DATA FROM INFLUXDB #################
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
 #             '\'r:p:lbnl:r:21954442-d84f76af LBNL Weather Station Dry-Bulb Temperature Sensor (Main, Two-Meter) (°F)\'']
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
	# 		time_period=["2017-11", "2018-06", None, None, None, None]
	# 	)

	# cleaned_data = main_obj.clean_data(df)
	# preprocessed_data = main_obj.preprocess_data(cleaned_data)
	# main_obj.model(preprocessed_data)

