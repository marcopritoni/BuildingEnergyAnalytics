'''
Last modified: July 11 2018
@author Pranav Gupta <phgupta@ucdavis.edu>
'''

import os
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

class Model_Data:

	def __init__(self, df):
		''' Constructor '''
		self.original_data = df
		
		self.baseline_period_in = pd.DataFrame()
		self.baseline_period_out = pd.DataFrame()
		self.year1_in = pd.DataFrame()
		self.year1_out = pd.DataFrame()
		self.year2_in = pd.DataFrame()
		self.year2_out = pd.DataFrame()

		self.predictions_train = pd.DataFrame()


	def split_data(self, time_period=None, input_col=None, output_col=None):
		''' 
		CHECK: time_period length is always going to be 6?
		This function converts time_periods into slice objects and split data (different from train_test_split) 
		'''

		if not isinstance(time_period, list):
			print("Error: time_period should be a list of strings")
			os._exit(1)
		elif len(time_period) != 6:
			print("Error: time_period length should be 6")
			os._exit(1)
		else:
			# Baseline period 1 
			tPeriod1 = (slice(time_period[0], time_period[1]))
			# Evaluation period 2 
			tPeriod2 = (slice(time_period[2], time_period[3]))
			# Evaluation period 3 
			tPeriod3 = (slice(time_period[4], time_period[5]))

			# Use all columns in the dataframe as the input variables
			if not input_col:
				input_col = self.original_data.columns

			try:
				self.baseline_period_in = self.original_data.loc[tPeriod1, input_col]
				self.baseline_period_out = self.original_data.loc[tPeriod1, output_col]
			except:
				print("Error: Could not retrieve baseline_period data")
				os._exit(1)

			try:
				self.year1_in = self.original_data.loc[tPeriod2, input_col]
				self.year1_out = self.original_data.loc[tPeriod2, output_col]
			except:
				print("Error: Could not retrieve Year 1 data")
				os._exit(1)

			try:
				self.year2_in = self.original_data.loc[tPeriod3, input_col]
				self.year2_out = self.original_data.loc[tPeriod3, output_col]
			except:
				print("Error: Could not retrieve Year 2 data")
				os._exit(1)