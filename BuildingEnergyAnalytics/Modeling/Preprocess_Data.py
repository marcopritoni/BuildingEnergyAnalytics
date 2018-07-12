import pandas as pd

class Preprocess_Data:

	def __init__(self, df):
		''' Constructor '''
		self.original_data = df
		self.preprocessed_data = pd.DataFrame()


	def add_degree_days(self, col='OAT', hdh_cpoint=65, cdh_cpoint=65):

		if self.preprocessed_data.empty:
			data = self.original_data
		else:
			data = self.preprocessed_data

		# Calculate hdh
		data['hdh'] = data[col]
		over_hdh = data.loc[:, col] > hdh_cpoint
		data.loc[over_hdh, 'hdh'] = 0
		data.loc[~over_hdh, 'hdh'] = hdh_cpoint - data.loc[~over_hdh, col]

		# Calculate cdh
		data['cdh'] = data[col]
		under_cdh = data.loc[:, col] < cdh_cpoint
		data.loc[under_cdh, 'cdh'] = 0
		data.loc[~under_cdh, 'cdh'] = data.loc[~under_cdh, col] - cdh_cpoint

		self.preprocessed_data = data


	def add_col_features(self, input_col=None, degree=None):
		''' Square/Cube specific input columns '''
		if input_col and degree:
			if len(input_col) != len(degree):
				print("Error: input_col and degree should have equal length")
			else:
				if self.preprocessed_data.empty:
					data = self.original_data
				else:
					data = self.preprocessed_data

				for i in range(len(input_col)):
					data.loc[:,input_col[i]+str(degree[i])] = pow(data.loc[:,input_col[i]],degree[i]) / pow(10,degree[i]-1)
				
				self.preprocessed_data = data


	def add_time_features(self, YEAR=False, MONTH=False, WEEK=True, TOD=True, DOW=True, DOY=False):
		'''
		# CHECK: Always add below time_feature or set it as parameter?
		# data["date"]=data.index.date
		'''

		if self.preprocessed_data.empty:
			data = self.original_data
		else:
			data = self.preprocessed_data

		if YEAR:
			data["YEAR"] = data.index.year
		if MONTH:
			data["MONTH"] = data.index.month
		if WEEK:
			data["WEEK"] = data.index.week
		if TOD:
			data["TOD"] = data.index.hour
		if DOW:
			data["DOW"] = data.index.weekday
		if DOY:
			data["DOY"] = data.index.dayofyear

		self.preprocessed_data = data


	def create_dummies(self, var_to_expand=['TOD','DOW', 'WEEK']):
		''' 
			CHECK: Drop last column to remove multi-collinearity.
		''' 

		if self.preprocessed_data.empty:
			data = self.original_data
		else: 
			data = self.preprocessed_data

		for var in var_to_expand:
			add_var = pd.get_dummies(data[var], prefix=var)
			# Add all the columns to the model data
			data = data.join(add_var)

		self.preprocessed_data = data