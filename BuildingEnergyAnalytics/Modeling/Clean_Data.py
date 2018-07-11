'''
Questions:
1. Are all the TODO's in Data_Preprocessor.py completed?
2. flag_data() and its sub-functions required?

Last modified: July 11 2018
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

	def rename_columns(self, col):
		try:
			self.cleaned_data.columns = col
		except:
			print("Error: Could not rename columns of dataframe!")

	def resample_data(self, data, freq):
		'''
			1. Also need to deal with energy quantities where resampling is .sum()
			2. Figure out how to apply different functions to different columns .apply()
			3. This theoretically work in upsampling too, check docs
			http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.resample.html 
		'''
		data = data.resample(freq).mean()
		return data

	def interpolate_data(self, data, limit):
		data = data.interpolate(how="index", limit=limit)
		return data

	def remove_na(self, data, remove_na_how):
		return data.dropna(how=remove_na_how)

	def remove_outliers(self, data, sd_val):
		'''
			this removes all data data above or below n sd_val from the mean
			this also excludes all lines with NA in any column
		'''
		data = data.dropna()
		data = data[(np.abs(stats.zscore(data)) < float(sd_val)).all(axis=1)]
		
		# CHECK: Missing code for below comment?
		# should do this with a moving windows
		
		return data

	def remove_out_of_bounds(self, data, low_bound, high_bound):
		data = data.dropna()
		data = data[(data > low_bound).all(axis=1) & (data < high_bound).all(axis=1)]
		
		# CHECK: Missing code for below comment?
		# this may need a different boundary for each column

		return data


	def clean_data(self, resample=True, freq='h', interpolate=True, limit=1,
					remove_na=True, remove_na_how='any', remove_outliers=True,
					sd_val=3, remove_out_of_bounds=True, low_bound=0, high_bound=9998):
		''' Clean dataframe '''

		data = self.original_data

		# CHECK: resample_data() results in duplicated rows
		if resample:
			try:
				data = self.resample_data(data, freq)
				print("Data resampled at \'%s\'" % freq)
			except:
				print("Error: Could not resample data")

		if interpolate:
			try:
				data = self.interpolate_data(data, limit=limit)
				print("Data interpolated with limit of %s element" % limit)
			except:
				print("Error: Could not interpolate data")
		
		if remove_na:
			try:
				data = self.remove_na(data, remove_na_how)
				print("Data NA removed")
			except:
				print("Error: Could not remove NA in data")

		if remove_outliers:
			try:
				data = self.remove_outliers(data, sd_val)
				print("Data outliers removed")
			except:
				print("Error: Could not remove data outliers")

		if remove_out_of_bounds:
			try:
				data = self.remove_out_of_bounds(data, low_bound, high_bound)
				print("Data out of bound points removed")
			except:
				print("Error: Could not remove data out of bound points")

		self.cleaned_data = data