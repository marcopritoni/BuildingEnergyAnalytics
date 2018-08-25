''' 

This file imports data from csv files and returns a dataframe. 

Notes
1. If only folder is specified and no filename, all csv's will be read in sorted order (by name)

To Do
1. Import data from InfluxDB, MongoDB and SkySpark.
2. Handle cases when user provides,
    1. file_name of type str and folder_name of type list(str)
    2. file_name and folder_name both of type list(str)

Last modified: August 24 2018
@author Marco Pritoni <marco.pritoni@gmail.com>
@author Jacob Rodriguez  <jbrodriguez@ucdavis.edu>
@author Pranav Gupta <phgupta@ucdavis.edu>

'''

import os
import glob
import numpy as np
import pandas as pd


class Import_Data:

    def __init__(self):
        ''' Constructor '''
        self.data = pd.DataFrame()


    def import_csv(self, file_name='*', folder_name='.', head_row=0, index_col=0, convert_col=True, concat_files=False):
        '''
            Import(s) csv file(s), append(s) to dataframe and returns it
            Note:
                1. If folder exists out of current directory, folder_name should contain correct regex
                2. Assuming there's no file called "*.csv"

        '''

        if not file_name and not folder_name:
            raise SystemError('Provide either file name or folder name.')

        # Import single csv file
        if isinstance(file_name, str) and isinstance(folder_name, str):
            try:
                self.data = self._load_csv(file_name, folder_name, head_row, index_col, convert_col, concat_files)
            except Exception as e:
                raise e

        # Import multiple csv files in a particular folder
        elif isinstance(file_name, list) and isinstance(folder_name, str):

            for i, file in enumerate(file_name):
                if isinstance(head_row, list):
                    _head_row = head_row[i]
                else:
                    _head_row = head_row

                if isinstance(index_col, list):
                    _index_col = index_col[i]
                else:
                    _index_col = index_col

                try:
                    data_tmp = self._load_csv(file, folder_name, _head_row, _index_col, convert_col, concat_files)
                    if concat_files:
                        self.data = self.data.append(data_tmp, ignore_index=False, verify_integrity=False)
                    else:
                        self.data = self.data.join(data_tmp, how="outer")
                except Exception as e:
                    raise e

        else:
            # Current implementation can't accept,
            # 1. file_name of type str and folder_name of type list(str)
            # 2. file_name and folder_name both of type list(str)
            raise SystemError


    def _load_csv(self, file_name, folder_name, head_row, index_col, convert_col, concat_files):
        ''' Load single csv file '''

        # Denotes all csv files
        if file_name == "*":

            if not os.path.isdir(folder_name):
                raise SystemError('Folder does not exist.')
            else:
                file_name_list = sorted(glob.glob(folder_name + '*.csv'))

                if not file_name_list:
                    raise SystemError('Either the folder does not contain any csv files or invalid folder provided.')
                else:
                    # Call previous function again with parameters changed (file_name=file_name_list, folder_name=None)
                    # Done to reduce redundancy of code
                    self.import_csv(file_name=file_name_list, head_row=head_row, index_col=index_col,
                                    convert_col=convert_col, concat_files=concat_files)
                    return self.data

        else:
            if not os.path.isdir(folder_name):
                raise SystemError('Folder does not exist.')
            else:
                path = os.path.join(folder_name, file_name)

                if head_row > 0:
                    data = pd.read_csv(path, index_col=index_col, skiprows=[i for i in range(head_row-1)])
                else:
                    data = pd.read_csv(path, index_col=index_col)

                # Convert time into datetime format
                try:
                    # Special case format 1/4/14 21:30
                    data.index = pd.to_datetime(data.index, format='%m/%d/%y %H:%M')
                except:
                    data.index = pd.to_datetime(data.index, dayfirst=False, infer_datetime_format=True)

        # Convert all columns to numeric type
        if convert_col:
            # Check columns in dataframe to see if they are numeric
            for col in data.columns:
                # If particular column is not numeric, then convert to numeric type
                if data[col].dtype != np.number:
                    data[col] = pd.to_numeric(data[col], errors="coerce")

        return data
