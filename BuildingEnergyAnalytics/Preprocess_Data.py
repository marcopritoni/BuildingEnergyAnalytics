'''

This file preprocesses the data by adding degree days, time features to capture seasonal/monthly... trends

Last modified: August 25 2018
@author Marco Pritoni <marco.pritoni@gmail.com>
@author Pranav Gupta <phgupta@ucdavis.edu>

'''

import pandas as pd

class Preprocess_Data:

    def __init__(self, df):
        ''' Constructor '''
        self.original_data = df
        self.preprocessed_data = pd.DataFrame()


    def add_degree_days(self, col='OAT', hdh_cpoint=65, cdh_cpoint=65):
        ''' Add Heating & Cooling Degree Hours. '''

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


    def add_col_features(self, col=None, degree=None):
        ''' Raise columns to the power of degree. '''

        if not col and not degree:
            return

        else:
            if isinstance(col, list) and isinstance(degree, list):
                if len(col) != len(degree):
                    raise SystemError('col and degree should have equal length.')
                else:
                    if self.preprocessed_data.empty:
                        data = self.original_data
                    else:
                        data = self.preprocessed_data

                    for i in range(len(col)):
                        data.loc[:,col[i]+str(degree[i])] = pow(data.loc[:,col[i]],degree[i]) / pow(10,degree[i]-1)
                    
                    self.preprocessed_data = data
            else:
                raise SystemError('col and degree should be lists.')


    def add_time_features(self, year=False, month=False, week=True, tod=True, dow=True, doy=False):
        ''' Add time features. '''

        if self.preprocessed_data.empty:
            data = self.original_data
        else:
            data = self.preprocessed_data

        if year:
            data["year"] = data.index.year
        if month:
            data["month"] = data.index.month
        if week:
            data["week"] = data.index.week
        if tod:
            data["tod"] = data.index.hour
        if dow:
            data["dow"] = data.index.weekday
        if doy:
            data["doy"] = data.index.dayofyear

        self.preprocessed_data = data


    def create_dummies(self, var_to_expand=None):
        ''' 
            One-hot encode time features. 
            e.g. var_to_expand=['tod', 'dow', 'year']
        '''

        if not var_to_expand:
            return

        else:
            if self.preprocessed_data.empty:
                data = self.original_data
            else: 
                data = self.preprocessed_data

            for var in var_to_expand:
                
                add_var = pd.get_dummies(data[var], prefix=var)
                
                # Add all the columns to the model data
                data = data.join(add_var)

                # Drop the original column that was expanded
                data.drop(columns=[var], inplace=True)

                # Drop last column to remove multi-collinearity
                cols = [col for col in data.columns if var in col]
                data.drop(columns=[cols[-1]], inplace=True)

            self.preprocessed_data = data
