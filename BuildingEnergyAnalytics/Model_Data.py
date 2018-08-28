'''

This file splits the data into baseline and projection periods, runs models on them and displays metrics & plots.

Last modified: August 25 2018
@author Pranav Gupta <phgupta@ucdavis.edu>

'''

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV, Lasso, Ridge, ElasticNet
from sklearn.model_selection import KFold, cross_val_score, train_test_split


class Model_Data:

    def __init__(self, df, input_col, output_col, time_period, exclude_time_period, alphas, cv, global_count):
        ''' Constructor '''

        self.original_data = df
        self.cv = cv
        self.global_count = global_count
        
        if (len(time_period) % 2 != 0) or (len(exclude_time_period) % 2 != 0):
            raise SystemError('time_periods need to be a multiple of 2 (i.e. have a start and end date)')
        else:
            self.time_period = time_period
            self.exclude_time_period = exclude_time_period

        if not output_col:
            raise SystemError('Please provide the target column.')
        elif not isinstance(output_col, str):
            raise SystemError('Target column should be a string.')
        else:
            self.output_col = output_col

        # Using all columns in dataframe as input cols
        if not input_col:
            input_col = list(self.original_data.columns)
            input_col.remove(output_col)
            self.input_col = input_col
        else:
            self.input_col = input_col

        if not isinstance(alphas, list) and not isinstance(alphas, np.ndarray):
            raise SystemError('alphas should be a list of int\'s or numpy ndarray.')
        else:
            self.alphas = alphas

        self.model          = None
        self.baseline_in    = pd.DataFrame()
        self.baseline_out   = pd.DataFrame()
        self.y_true         = pd.DataFrame()
        self.y_pred         = pd.DataFrame()
        self.metrics        = {}

        self.models         = []
        self.model_names    = []
        self.max_scores     = []
        self.alpha_scores   = []


    def split_data(self):
        ''' Split data according to time_period values '''

        time_period1 = (slice(self.time_period[0], self.time_period[1]))
        exclude_time_period1 = (slice(self.exclude_time_period[0], self.exclude_time_period[1]))
        try:
            # Extract data ranging in time_period1
            self.baseline_in = self.original_data.loc[time_period1, self.input_col]
            self.baseline_out = self.original_data.loc[time_period1, self.output_col]

            if self.exclude_time_period[0] and self.exclude_time_period[1]:
                # Drop data ranging in exclude_time_period1
                self.baseline_in.drop(self.baseline_in.loc[exclude_time_period1].index, axis=0, inplace=True)
                self.baseline_out.drop(self.baseline_out.loc[exclude_time_period1].index, axis=0, inplace=True)
        except Exception as e:
            raise e

        # CHECK: Can optimize this part
        # Error checking to ensure time_period values are valid
        if len(self.time_period) > 2:
            for i in range(2, len(self.time_period), 2):
                period = (slice(self.time_period[i], self.time_period[i+1]))
                try:
                    self.original_data.loc[period, self.input_col]
                    self.original_data.loc[period, self.output_col]
                except Exception as e:
                    raise e


    def adj_r2(self, r2, n, k):
        ''' Return adjusted r2 score '''
        return 1 - (((1 - r2) * (n - 1)) / (n - k - 1))


    def linear_regression(self):
        ''' Linear Regression '''

        model = LinearRegression(normalize=True)
        scores = cross_val_score(model, self.baseline_in, self.baseline_out, cv=self.cv)
        mean_score = sum(scores) / len(scores)
        
        self.models.append(model)
        self.model_names.append('Linear Regression')
        self.max_scores.append(mean_score)


    def lasso_regression(self):
        ''' Lasso Regression '''

        score_list = []
        max_score = float('-inf')
        best_alpha = None

        for alpha in self.alphas:
            model = Lasso(normalize=True, alpha=alpha, max_iter=5000)
            model.fit(self.baseline_in, self.baseline_out.values.ravel())
            scores = cross_val_score(model, self.baseline_in, self.baseline_out, cv=self.cv)
            mean_score = np.mean(scores)
            score_list.append(mean_score)
            
            if mean_score > max_score:
                max_score = mean_score
                best_alpha = alpha

        adj_r2 = self.adj_r2(max_score, self.baseline_in.shape[0], self.baseline_in.shape[1])

        self.models.append(Lasso(alpha=best_alpha))
        self.model_names.append('Lasso Regression')
        self.max_scores.append(max_score)
        self.alpha_scores.append(score_list)


    def ridge_regression(self):
        ''' Ridge Regression '''

        score_list = []
        max_score = float('-inf')
        best_alpha = None

        for alpha in self.alphas:
            model = Ridge(normalize=True, alpha=alpha, max_iter=5000)
            model.fit(self.baseline_in, self.baseline_out.values.ravel())
            scores = cross_val_score(model, self.baseline_in, self.baseline_out, cv=self.cv)
            mean_score = np.mean(scores)
            score_list.append(mean_score)
            
            if mean_score > max_score:
                max_score = mean_score
                best_alpha = alpha

        adj_r2 = self.adj_r2(max_score, self.baseline_in.shape[0], self.baseline_in.shape[1])

        self.models.append(Ridge(alpha=best_alpha))
        self.model_names.append('Ridge Regression')
        self.max_scores.append(max_score)
        self.alpha_scores.append(score_list)


    def elastic_net_regression(self):
        ''' ElasticNet Regression '''

        score_list = []
        max_score = float('-inf')
        best_alpha = None

        for alpha in self.alphas:
            model = ElasticNet(normalize=True, alpha=alpha, max_iter=5000)
            model.fit(self.baseline_in, self.baseline_out.values.ravel())
            scores = cross_val_score(model, self.baseline_in, self.baseline_out, cv=self.cv)
            mean_score = np.mean(scores)
            score_list.append(mean_score)
            
            if mean_score > max_score:
                max_score = mean_score
                best_alpha = alpha

        adj_r2 = self.adj_r2(max_score, self.baseline_in.shape[0], self.baseline_in.shape[1])

        self.models.append(ElasticNet(alpha=best_alpha))
        self.model_names.append('ElasticNet Regression')
        self.max_scores.append(max_score)
        self.alpha_scores.append(score_list)


    def run_models(self):
        ''' Run all models and find optimal model '''

        self.linear_regression()
        self.lasso_regression()
        self.ridge_regression()
        self.elastic_net_regression()

        # Find model with maximum score
        max_score = max(self.max_scores)

        return self.models[self.max_scores.index(max_score)], self.model_names[self.max_scores.index(max_score)]


    def custom_model(self, func):
        '''
            TODO: Define custom function's parameters, its data types, and return types
        '''

        y_pred = func(self.baseline_in, self.baseline_out)

        self.custom_metrics = {}
        self.custom_metrics['r2'] = r2_score(self.baseline_out, y_pred)
        self.custom_metrics['mse'] = mean_squared_error(self.baseline_out, y_pred)
        self.custom_metrics['rmse'] = math.sqrt(self.custom_metrics['mse'])
        self.custom_metrics['adj_r2'] = self.adj_r2(self.custom_metrics['r2'], self.baseline_in.shape[0], self.baseline_in.shape[1])

        return self.custom_metrics


    def best_model_fit(self, model):
        ''' Fit data to optimal model '''

        self.model = model
        X_train, X_test, y_train, y_test = train_test_split(self.baseline_in, self.baseline_out, 
                                                        test_size=0.30, random_state=42)

        self.model.fit(X_train, y_train)
        self.y_true = y_test # Pandas Series
        self.y_pred = self.model.predict(X_test) # numpy.ndarray
        
        # n and k values for adj r2 score
        self.n_test = X_test.shape[0]
        self.k_test = X_test.shape[1]


    def display_metrics(self):
        ''' Display metrics '''

        self.metrics['r2'] = r2_score(self.y_true, self.y_pred)
        self.metrics['mse'] = mean_squared_error(self.y_true, self.y_pred)
        self.metrics['rmse'] = math.sqrt(self.metrics['mse'])
        self.metrics['adj_r2'] = self.adj_r2(self.metrics['r2'], self.n_test, self.k_test) # DOUBLE CHECK.

        return self.metrics


    def display_plots(self, figsize):

        # Figure 1
        # Plot Model Score vs Alphas to get an idea of which alphas work best
        # fig1 = plt.figure(1)
        fig1 = plt.figure(self.global_count)
        plt.plot(self.alphas, self.alpha_scores[0], color='blue', label=self.model_names[0])
        plt.plot(self.alphas, self.alpha_scores[1], color='black', label=self.model_names[1])
        plt.plot(self.alphas, self.alpha_scores[2], color='red', label=self.model_names[2])
        plt.xlabel('Alphas')
        plt.ylabel('Model Accuracy')
        plt.title("R2 Score v/s alpha")
        plt.legend()


        # Figure 2
        # Baseline and projection plots
        # fig2 = plt.figure(2)
        fig2 = plt.figure(self.global_count + 1)

        # Number of plots to display
        nrows = len(self.time_period) / 2
        
        # Plot 1 - Baseline
        base_df = pd.DataFrame()
        base_df['y_true'] = self.y_true
        base_df['y_pred'] = self.y_pred
        ax1 = fig2.add_subplot(nrows, 1, 1)
        base_df.plot(ax=ax1, figsize=figsize,
            title='Baseline Period ({}-{})'.format(self.time_period[0], self.time_period[1]))

        # Display projection plots
        if len(self.time_period) > 2:
            
            num_plot = 2
            for i in range(2, len(self.time_period), 2):
                ax = fig2.add_subplot(nrows, 1, num_plot)
                period = (slice(self.time_period[i], self.time_period[i+1]))
                project_df = pd.DataFrame()
                project_df['y_true'] = self.original_data.loc[period, self.output_col]
                project_df['y_pred'] = self.model.predict(self.original_data.loc[period, self.input_col])
                project_df.plot(ax=ax, figsize=figsize,
                    title='Projection Period ({}-{})'.format(self.time_period[i], self.time_period[i+1]))
                num_plot += 1

        fig2.tight_layout()

        return fig1, fig2
