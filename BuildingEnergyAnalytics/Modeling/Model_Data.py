'''
Last modified: July 11 2018
@author Pranav Gupta <phgupta@ucdavis.edu>
'''

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

class Model_Data:

	def __init__(self, df, model, input_col=None, output_col=None):
		''' Constructor '''
		self.original_data = df
		self.model = model

		if not input_col:
			input_col = list(self.original_data.columns)
			input_col.remove(output_col)
			self.input_col = input_col
		else:
			self.input_col = input_col

		self.output_col = output_col
		
		self.baseline_period_in = pd.DataFrame()
		self.baseline_period_out = pd.DataFrame()
		self.year1_in = pd.DataFrame()
		self.year1_out = pd.DataFrame()
		self.year2_in = pd.DataFrame()
		self.year2_out = pd.DataFrame()

		self.metrics = {}


	def split_data(self, time_period=None):
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

			try:
				self.baseline_period_in = self.original_data.loc[tPeriod1, self.input_col]
				self.baseline_period_out = self.original_data.loc[tPeriod1, self.output_col]
			except:
				print("Error: Could not retrieve baseline_period data")
				os._exit(1)

			try:
				self.year1_in = self.original_data.loc[tPeriod2, self.input_col]
				self.year1_out = self.original_data.loc[tPeriod2, self.output_col]
			except:
				print("Error: Could not retrieve Year 1 data")
				os._exit(1)

			try:
				self.year2_in = self.original_data.loc[tPeriod3, self.input_col]
				self.year2_out = self.original_data.loc[tPeriod3, self.output_col]
			except:
				print("Error: Could not retrieve Year 2 data")
				os._exit(1)


	def model_fit(self):

		self.metrics["Cross_Val"] = cross_val_score(self.model, self.baseline_period_in.dropna(), 
													self.baseline_period_out.dropna())


	def display_plots(self):

		# Creating dataframe
		plot_df = pd.DataFrame()
		plot_df["y_true"] = self.baseline_period_out.dropna()
		self.model.fit(self.baseline_period_in.dropna(), self.baseline_period_out.dropna())
		plot_df["y_pred"] = self.model.predict(self.baseline_period_in.dropna())
		# plot_df.plot(figsize=(15,5)) # CHECK: Figure out how to plot a dataframe
		# Scatter plot
		plt.scatter(plot_df['y_true'], plot_df['y_pred'])
		plt.xlabel("True Values")
		plt.ylabel("Predicted Values")
		plt.title("Baseline Period [{}]".format(self.output_col))
		plt.show()
		
		# Project to Year 1
		plot_df1 = pd.DataFrame()
		plot_df1["y_true"] = self.year1_out.dropna()
		plot_df1["y_pred"] = self.model.predict(self.year1_in.dropna())
		sav_plt = plot_df1.diff(axis=1)["y_pred"]
		# Plot projection vs real post data
		plot_df1.plot(figsize=(15,5))
		print("Cumulative savings = %f percent" % ((sav_plt.sum()/(plot_df1[["y_pred"]].sum()[0])*100)))

		# Project to Year 2
		plot_df2 = pd.DataFrame()
		plot_df2["y_true"] = self.year2_out.dropna()
		plot_df2["y_pred"] = self.model.predict(self.year2_in.dropna())
		sav_plt = plot_df2.diff(axis=1)["y_pred"]
		# Plot projection vs real post data
		plot_df2.plot(figsize=(15,5))
		print("Cumulative savings = %f percent" % ((sav_plt.sum()/(plot_df2[["y_pred"]].sum()[0])*100)))


	def calc_metrics(self):
		pass
