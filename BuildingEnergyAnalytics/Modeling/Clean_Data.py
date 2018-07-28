'''
Questions:
1. Are all the TODO's in Data_Preprocessor.py completed?
2. flag_data() and its sub-functions required?

Last modified: July 28 2018
@author Marco Pritoni <marco.pritoni@gmail.com>
'''

import os
import numpy as np
import pandas as pd
from scipy import stats

class Clean_Data:

	def __init__(self, df, f):
		''' Constructor '''
		self.original_data = df
		self.cleaned_data = pd.DataFrame()
		self.f = f

	def rename_columns(self, col):
		try:
			self.cleaned_data.columns = col
		except:
			# print("Error: Could not rename columns of dataframe!")
			self.f.write('Error: Could not rename columns of dataframe!\n')
			os._exit(1)


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
				# print("Data resampled at \'%s\'" % freq)
				self.f.write('Data resampled at \'{}\'\n'.format(freq))
			except:
				# print("Error: Could not resample data")
				self.f.write('Error: Could not resample data\n')
				os._exit(1)

		if interpolate:
			try:
				data = self.interpolate_data(data, limit=limit)
				# print("Data interpolated with limit of %s element" % limit)
				self.f.write('Data interpolated with limit of {} element\n'.format(limit))
			except:
				# print("Error: Could not interpolate data")
				self.f.write('Error: Could not interpolate data\n')
				os._exit(1)
		
		if remove_na:
			try:
				data = self.remove_na(data, remove_na_how)
				# print("Data NA removed")
				self.f.write('Data NA removed\n')
			except:
				# print("Error: Could not remove NA in data")
				self.f.write('Error: Could not remove NA in data\n')
				os._exit(1)

		if remove_outliers:
			try:
				data = self.remove_outliers(data, sd_val)
				# print("Data outliers removed")
				self.f.write('Data outliers removed\n')
			except:
				# print("Error: Could not remove data outliers")
				self.f.write('Error: Could not remove data outliers\n')
				os._exit(1)

		if remove_out_of_bounds:
			try:
				data = self.remove_out_of_bounds(data, low_bound, high_bound)
				# print("Data out of bound points removed")
				self.f.write('Data out of bound points removed\n')
			except:
				# print("Error: Could not remove data out of bound points")
				self.f.write('Error: Could not remove data out of bound points\n')
				os._exit(1)

		self.cleaned_data = data