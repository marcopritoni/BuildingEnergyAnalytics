'''
Last modified: July 28 2018
@author Pranav Gupta <phgupta@ucdavis.edu>
'''

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.model_selection import KFold, cross_val_score, train_test_split

class Model_Data:

    def __init__(self, df, time_period, output_col, input_col=None):
        ''' Constructor '''
        self.original_data = df
        self.output_col = output_col

        if (not time_period) or (len(time_period) % 2 != 0):
            print("Error: time_period needs to be a multiple of 2 (i.e. have a start and end date)")
            os._exit(1)
        else:
            self.time_period = time_period

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

        self.scores = []
        self.models = []
        self.model_names = []


    def split_data(self):
        ''' Split data according to time_period values '''

        time_period1 = (slice(self.time_period[0], self.time_period[1]))
        try:
            self.baseline_period_in = self.original_data.loc[time_period1, self.input_col]
            self.baseline_period_out = self.original_data.loc[time_period1, self.output_col]
        except:
            print("Error: Could not retrieve baseline_period data")
            os._exit(1)

        # Error checking to ensure time_period values are valid
        if len(self.time_period) > 2:
            for i in range(2, len(self.time_period), 2):
                period = (slice(self.time_period[i], self.time_period[i+1]))
                try:
                    self.original_data.loc[period, self.input_col]
                    self.original_data.loc[period, self.output_col]
                except:
                    print("Error: Could not retrieve projection period data")
                    os._exit(1)


    def linear_regression(self):

        print("Linear Regression...")
        model = LinearRegression()
        scores = cross_val_score(model, self.baseline_period_in, self.baseline_period_out)
        mean_score = sum(scores) / len(scores)
        
        print("Cross Val Scores: ", scores)
        print("Mean Cross Val Score: ", mean_score)

        self.models.append(model)
        self.scores.append(mean_score)
        self.model_names.append('Linear Regression')


    def lasso_regression(self):

        print("Lasso Regression...")
        model = LassoCV()
        model.fit(self.baseline_period_in, self.baseline_period_out.values.ravel())
        score = model.score(self.baseline_period_in, self.baseline_period_out)
        adj_r2 = 1 - ((1-score)*(self.baseline_period_in.shape[0]-1) / (self.baseline_period_in.shape[0]-self.baseline_period_in.shape[1]-1))
        # print("Grid of alphas used for fitting: \n", model.alphas_)
        print("Best Alpha: ", model.alpha_)
        print("Mean Cross Val Score: ", score)
        print("Adj R2 Score: ", adj_r2)

        self.models.append(model)
        self.scores.append(score)
        self.model_names.append('Lasso Regression')


    def ridge_regression(self):

        print("Ridge Regression...")
        model = RidgeCV()
        model.fit(self.baseline_period_in, self.baseline_period_out.values.ravel())
        score = model.score(self.baseline_period_in, self.baseline_period_out)
        adj_r2 = 1 - ((1-score)*(self.baseline_period_in.shape[0]-1) / (self.baseline_period_in.shape[0]-self.baseline_period_in.shape[1]-1))
        # print("Grid of alphas used for fitting: \n", model.alphas_)
        print("Best Alpha: ", model.alpha_)
        print("Mean Cross Val Score: ", score)
        print("Adj R2 Score: ", adj_r2)

        self.models.append(model)
        self.scores.append(score)
        self.model_names.append('Ridge Regression')


    def elastic_net_regression(self):

        print("Elastic Net Regression...")
        model = ElasticNetCV()
        model.fit(self.baseline_period_in, self.baseline_period_out.values.ravel())
        score = model.score(self.baseline_period_in, self.baseline_period_out)
        adj_r2 = 1 - ((1-score)*(self.baseline_period_in.shape[0]-1) / (self.baseline_period_in.shape[0]-self.baseline_period_in.shape[1]-1))
        # print("Grid of alphas used for fitting: \n", model.alphas_)
        print("Best Alpha: ", model.alpha_)
        print("Mean Cross Val Score: ", score)
        print("Adj R2 Score: ", adj_r2)

        self.models.append(model)
        self.scores.append(score)
        self.model_names.append('Elastic Net Regression')


    def run_models(self):

        self.linear_regression()
        self.lasso_regression()
        self.ridge_regression()
        self.elastic_net_regression()

        max_score = float('-inf')
        for _, score, _ in zip(self.models, self.scores, self.model_names):
            if score > max_score:
                max_score = score

        print('Choosing model with highest score: ', self.model_names[self.scores.index(max_score)])
        return self.models[self.scores.index(max_score)], self.model_names[self.scores.index(max_score)]


    def best_model_fit(self, model):

        self.model = model

        X_train, X_test, y_train, y_test = train_test_split(self.baseline_period_in, self.baseline_period_out, 
                                                        test_size=0.30, random_state=42)

        self.model.fit(X_train, y_train)
        self.y_true = y_test # Pandas Series
        self.y_pred = self.model.predict(X_test) # Array


    def display_metrics(self):

        r2 = r2_score(self.y_true, self.y_pred)
        mse = mean_squared_error(self.y_true, self.y_pred)
        adj_r2 = 1 - (1 - r2) * (self.y_true.count() - 1) / (self.y_true.count() - len(self.input_col) - 1)

        print('{:<10}: {}'.format('R2', r2))
        print('{:<10}: {}'.format('MSE', mse))
        print('{:<10}: {}'.format('Adj_R2', adj_r2))


    def display_plots(self):

        # Number of plots to display
        nrows = len(self.time_period) / 2
        
        # Create figure
        fig = plt.figure(1)

        # Plot 1 - Baseline
        base_df = pd.DataFrame()
        base_df['y_true'] = self.y_true
        base_df['y_pred'] = self.y_pred
        ax1 = fig.add_subplot(nrows, 1, 1)
        base_df.plot(ax=ax1, title='Baseline Period ({}-{})'.format(self.time_period[0], self.time_period[1]))

        # Display projection plots
        if len(self.time_period) > 2:
            
            num_plot = 2
            for i in range(2, len(self.time_period), 2):
                ax = fig.add_subplot(nrows, 1, num_plot)
                period = (slice(self.time_period[i], self.time_period[i+1]))
                project_df = pd.DataFrame()
                project_df['y_true'] = self.original_data.loc[period, self.output_col]
                project_df['y_pred'] = self.model.predict(self.original_data.loc[period, self.input_col])
                project_df.plot(ax=ax, title='Projection Period ({}-{})'.format(self.time_period[i], self.time_period[i+1]))
                num_plot += 1

        plt.show()

