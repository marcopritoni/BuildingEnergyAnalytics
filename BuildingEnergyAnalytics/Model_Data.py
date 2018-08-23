'''
Last modified: August 23 2018
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

    def __init__(self, df, time_period, exclude_time_period, output_col, alphas, input_col):
        ''' Constructor '''
        self.original_data = df
        self.output_col = output_col
        # self.f = f
        self.alphas = alphas

        if (len(time_period) % 2 != 0) or (len(exclude_time_period) % 2 != 0):
            # print("Error: time_period needs to be a multiple of 2 (i.e. have a start and end date)")
            # self.f.write('Error: time_period needs to be a multiple of 2 (i.e. have a start and end date)\n')
            raise SystemError('time_period needs to be a multiple of 2 (i.e. have a start and end date)')
        else:
            self.time_period = time_period
            self.exclude_time_period = exclude_time_period

        if not input_col:
            input_col = list(self.original_data.columns)
            input_col.remove(output_col)
            self.input_col = input_col
        else:
            self.input_col = input_col

        self.model = None
        self.baseline_period_in = pd.DataFrame()
        self.baseline_period_out = pd.DataFrame()
        self.y_true = pd.DataFrame()
        self.y_pred = pd.DataFrame()
        self.metrics = {}

        self.score_list = []    # CHECK: Change variable names
        self.scores = []
        self.models = []
        self.model_names = []


    def split_data(self):
        ''' Split data according to time_period values '''

        time_period1 = (slice(self.time_period[0], self.time_period[1]))
        exclude_time_period1 = (slice(self.exclude_time_period[0], self.exclude_time_period[1]))
        try:
            # Extract data ranging in time_period1
            self.baseline_period_in = self.original_data.loc[time_period1, self.input_col]
            self.baseline_period_out = self.original_data.loc[time_period1, self.output_col]

            if self.exclude_time_period[0] and self.exclude_time_period[1]:
                # Drop data ranging in exclude_time_period1
                self.baseline_period_in.drop(self.baseline_period_in.loc[exclude_time_period1].index, axis=0, inplace=True)
                self.baseline_period_out.drop(self.baseline_period_out.loc[exclude_time_period1].index, axis=0, inplace=True)
        except Exception as e:
            raise

        # Error checking to ensure time_period values are valid
        if len(self.time_period) > 2:
            for i in range(2, len(self.time_period), 2):
                period = (slice(self.time_period[i], self.time_period[i+1]))
                try:
                    self.original_data.loc[period, self.input_col]
                    self.original_data.loc[period, self.output_col]
                except Exception as e:
                    raise


    def linear_regression(self):

        model = LinearRegression(normalize=True)
        scores = cross_val_score(model, self.baseline_period_in, self.baseline_period_out)
        mean_score = sum(scores) / len(scores)
        
        self.models.append(model)
        self.scores.append(mean_score)
        self.model_names.append('Linear Regression')


    def lasso_regression(self):

        score_list = []
        max_score = float('-inf')
        best_alpha = None

        for alpha in self.alphas:
            print('Lasso Regression: Alpha = ', alpha)
            model = Lasso(normalize=True, alpha=alpha, max_iter=5000)
            model.fit(self.baseline_period_in, self.baseline_period_out.values.ravel())
            scores = cross_val_score(model, self.baseline_period_in, self.baseline_period_out)
            mean_score = np.mean(scores)
            score_list.append(mean_score)
            
            if mean_score > max_score:
                max_score = mean_score
                best_alpha = alpha

        adj_r2 = 1 - ((1-max_score)*(self.baseline_period_in.shape[0]-1) / (self.baseline_period_in.shape[0]-self.baseline_period_in.shape[1]-1))
        
        self.models.append(Lasso(alpha=best_alpha))
        self.scores.append(max_score)
        self.model_names.append('Lasso Regression')
        self.score_list.append(score_list)


    def ridge_regression(self):

        score_list = []
        max_score = float('-inf')
        best_alpha = None

        for alpha in self.alphas:
            print('Ridge Regression: Alpha = ', alpha)
            model = Ridge(normalize=True, alpha=alpha, max_iter=5000)
            model.fit(self.baseline_period_in, self.baseline_period_out.values.ravel())
            scores = cross_val_score(model, self.baseline_period_in, self.baseline_period_out)
            mean_score = np.mean(scores)
            score_list.append(mean_score)
            
            if mean_score > max_score:
                max_score = mean_score
                best_alpha = alpha

        adj_r2 = 1 - ((1-max_score)*(self.baseline_period_in.shape[0]-1) / (self.baseline_period_in.shape[0]-self.baseline_period_in.shape[1]-1))

        self.models.append(Ridge(alpha=best_alpha))
        self.scores.append(max_score)
        self.model_names.append('Ridge Regression')
        self.score_list.append(score_list)


    def elastic_net_regression(self):

        score_list = []
        max_score = float('-inf')
        best_alpha = None

        for alpha in self.alphas:
            print('ElasticNet: Alpha = ', alpha)
            model = ElasticNet(normalize=True, alpha=alpha, max_iter=5000)
            model.fit(self.baseline_period_in, self.baseline_period_out.values.ravel())
            scores = cross_val_score(model, self.baseline_period_in, self.baseline_period_out)
            mean_score = np.mean(scores)
            score_list.append(mean_score)
            
            if mean_score > max_score:
                max_score = mean_score
                best_alpha = alpha

        adj_r2 = 1 - ((1-max_score)*(self.baseline_period_in.shape[0]-1) / (self.baseline_period_in.shape[0]-self.baseline_period_in.shape[1]-1))

        self.models.append(ElasticNet(alpha=best_alpha))
        self.scores.append(max_score)
        self.model_names.append('ElasticNet Regression')
        self.score_list.append(score_list)


    def run_models(self):

        self.linear_regression()
        self.lasso_regression()
        self.ridge_regression()
        self.elastic_net_regression()

        max_score = max(self.scores)

        return self.models[self.scores.index(max_score)], self.model_names[self.scores.index(max_score)]


    def custom_model(self, func):
        '''
            TODO: Define custom function's parameters, its data types, and return types
        '''

        y_pred = func(self.baseline_period_in, self.baseline_period_out)

        metrics = {}
        metrics['r2'] = r2_score(self.baseline_period_out, y_pred)
        metrics['mse'] = mean_squared_error(self.baseline_period_out, y_pred)
        metrics['rmse'] = math.sqrt(metrics['mse'])
        metrics['adj_r2'] = 1 - ((1 - metrics['r2']) * (self.baseline_period_out.count() - 1) / \
                            (self.baseline_period_out.count() - len(self.baseline_period_in.count()) - 1))


    def best_model_fit(self, model):

        self.model = model
        X_train, X_test, y_train, y_test = train_test_split(self.baseline_period_in, self.baseline_period_out, 
                                                        test_size=0.30, random_state=42)

        self.model.fit(X_train, y_train)
        self.y_true = y_test # Pandas Series
        self.y_pred = self.model.predict(X_test) # Array


    def display_metrics(self):

        self.metrics['r2'] = r2_score(self.y_true, self.y_pred)
        self.metrics['mse'] = mean_squared_error(self.y_true, self.y_pred)
        self.metrics['rmse'] = math.sqrt(self.metrics['mse'])
        self.metrics['adj_r2'] = 1 - ((1 - self.metrics['r2']) * (self.y_true.count() - 1) / (self.y_true.count() - len(self.input_col) - 1))

        return self.metrics


    def display_plots(self, figsize):

        # Figure 1 - Plot Model Score vs Alphas to get an idea of which alphas work best
        fig1 = plt.figure(1)
        plt.plot(self.alphas, self.score_list[0], color='blue', label=self.model_names[0])
        plt.plot(self.alphas, self.score_list[1], color='black', label=self.model_names[1])
        plt.plot(self.alphas, self.score_list[2], color='red', label=self.model_names[2])
        plt.xlabel('Alphas')
        plt.ylabel('Model Accuracy')
        plt.title("R2 Score v/s alpha")
        plt.legend()

        # Figure 2 - Baseline and projection plots
        fig2 = plt.figure(2)

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
