'''
Last modified: August 18 2018
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
            # print("Error: Could not retrieve baseline_period data")
            # self.f.write('Error: Could not retrieve baseline_period data\n')
            # self.f.write(str(e))
            raise

        # Error checking to ensure time_period values are valid
        if len(self.time_period) > 2:
            for i in range(2, len(self.time_period), 2):
                period = (slice(self.time_period[i], self.time_period[i+1]))
                try:
                    self.original_data.loc[period, self.input_col]
                    self.original_data.loc[period, self.output_col]
                except Exception as e:
                    # print("Error: Could not retrieve projection period data")
                    # self.f.write('Error: Could not retrieve projection period data\n')
                    # self.f.write(str(e))
                    raise


    def linear_regression(self):

        # print("Linear Regression...")
        # self.f.write('\nLinear Regression...\n')

        model = LinearRegression(normalize=True)
        scores = cross_val_score(model, self.baseline_period_in, self.baseline_period_out)
        mean_score = sum(scores) / len(scores)
        
        # print("Cross Val Scores: ", scores)
        # self.f.write('Cross Val Scores: {}\n'.format(scores))
        # print("Mean Cross Val Score: ", mean_score)
        # self.f.write('Mean Cross Val Scores: {}\n'.format(mean_score))

        self.models.append(model)
        self.scores.append(mean_score)
        self.model_names.append('Linear Regression')


    def lasso_regression(self):

        # print("Lasso Regression...")
        # self.f.write('\nLasso Regression...\n')

        score_list = []
        max_score = float('-inf')
        best_alpha = None

        for alpha in self.alphas:
            print('Lasso')
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

        # self.f.write('Best Alpha: {}\n'.format(best_alpha))
        # self.f.write('Cross Val Score: {}\n'.format(max_score))
        # self.f.write('Adj R2 Score: {}\n'.format(adj_r2))

        return score_list

        '''
        # print("Lasso Regression...")
        self.f.write('\nLasso Regression...\n')

        model = LassoCV(normalize=True)
        model.fit(self.baseline_period_in, self.baseline_period_out.values.ravel())
        score = model.score(self.baseline_period_in, self.baseline_period_out)
        adj_r2 = 1 - ((1-score)*(self.baseline_period_in.shape[0]-1) / (self.baseline_period_in.shape[0]-self.baseline_period_in.shape[1]-1))
        
        # print("Grid of alphas used for fitting: \n", model.alphas_)
        # print("Best Alpha: ", model.alpha_)
        self.f.write('Best Alpha: {}\n'.format(model.alpha_))
        # print("Mean Cross Val Score: ", score)
        self.f.write('Mean Cross Val Score: {}\n'.format(score))
        # print("Adj R2 Score: ", adj_r2)
        self.f.write('Adj R2 Score: {}\n'.format(adj_r2))

        self.models.append(model)
        self.scores.append(score)
        self.model_names.append('Lasso Regression')
        '''


    def ridge_regression(self):

        # print("Ridge Regression...")
        # self.f.write('\nRidge Regression...\n')

        score_list = []
        max_score = float('-inf')
        best_alpha = None

        for alpha in self.alphas:
            print('Ridge')
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

        # self.f.write('Best Alpha: {}\n'.format(best_alpha))
        # self.f.write('Cross Val Score: {}\n'.format(max_score))
        # self.f.write('Adj R2 Score: {}\n'.format(adj_r2))

        return score_list

        '''
        # print("Ridge Regression...")
        self.f.write('\nRidge Regression...\n')

        model = RidgeCV(normalize=True)
        model.fit(self.baseline_period_in, self.baseline_period_out.values.ravel())
        score = model.score(self.baseline_period_in, self.baseline_period_out)
        adj_r2 = 1 - ((1-score)*(self.baseline_period_in.shape[0]-1) / (self.baseline_period_in.shape[0]-self.baseline_period_in.shape[1]-1))
        
        # print("Grid of alphas used for fitting: \n", model.alphas_)
        # print("Best Alpha: ", model.alpha_)
        self.f.write('Best Alpha: {}\n'.format(model.alpha_))
        # print("Mean Cross Val Score: ", score)
        self.f.write('Mean Cross Val Score: {}\n'.format(score))
        # print("Adj R2 Score: ", adj_r2)
        self.f.write('Adj R2 Score: {}\n'.format(adj_r2))

        self.models.append(model)
        self.scores.append(score)
        self.model_names.append('Ridge Regression')
        '''


    def elastic_net_regression(self):

        # print("Lasso Regression...")
        # self.f.write('\nElastic Net Regression...\n')

        score_list = []
        max_score = float('-inf')
        best_alpha = None

        for alpha in self.alphas:
            print('ElasticNet')
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

        # self.f.write('Best Alpha: {}\n'.format(best_alpha))
        # self.f.write('Cross Val Score: {}\n'.format(max_score))
        # self.f.write('Adj R2 Score: {}\n'.format(adj_r2))

        return score_list

        '''
        # print("Elastic Net Regression...")
        self.f.write('\nElastic Net Regression...\n')

        model = ElasticNetCV(normalize=True)
        model.fit(self.baseline_period_in, self.baseline_period_out.values.ravel())
        score = model.score(self.baseline_period_in, self.baseline_period_out)
        adj_r2 = 1 - ((1-score)*(self.baseline_period_in.shape[0]-1) / (self.baseline_period_in.shape[0]-self.baseline_period_in.shape[1]-1))
        
        # print("Grid of alphas used for fitting: \n", model.alphas_)
        # print("Best Alpha: ", model.alpha_)
        self.f.write('Best Alpha: {}\n'.format(model.alpha_))
        # print("Mean Cross Val Score: ", score)
        self.f.write('Mean Cross Val Score: {}\n'.format(score))
        # print("Adj R2 Score: ", adj_r2)
        self.f.write('Adj R2 Score: {}\n'.format(adj_r2))

        self.models.append(model)
        self.scores.append(score)
        self.model_names.append('Elastic Net Regression')
        '''


    def run_models(self):

        self.linear_regression()
        lasso_scores = self.lasso_regression()
        ridge_scores = self.ridge_regression()
        elastic_scores = self.elastic_net_regression()

        # Plot Model Score vs Alphas to get an idea of which alphas work best
        fig1 = plt.figure(1)
        plt.plot(self.alphas, lasso_scores, color='blue', label='Lasso')
        plt.plot(self.alphas, ridge_scores, color='black', label='Ridge')
        plt.plot(self.alphas, elastic_scores, color='red', label='ElasticNet')
        plt.xlabel('Alpha')
        plt.ylabel('Model Accuracy')
        plt.title("R2 Score with varying alpha")
        plt.legend()
        # plt.show()
        # plt.savefig('acc_alpha.png')
        plt.savefig('acc_alpha.png')

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

        # print('{:<10}: {}'.format('R2', r2))
        # self.f.write('{:<10}: {}\n'.format('R2', metrics['r2']))
        # print('{:<10}: {}'.format('MSE', mse))
        # self.f.write('{:<10}: {}\n'.format('MSE', metrics['mse']))
        # print('{:<10}: {}'.format('MSE', mse))
        # self.f.write('{:<10}: {}\n'.format('RMSE', metrics['rmse']))
        # print('{:<10}: {}'.format('Adj_R2', adj_r2))
        # self.f.write('{:<10}: {}\n'.format('Adj_R2', metrics['adj_r2']))


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

        # print('{:<10}: {}'.format('R2', r2))
        # self.f.write('{:<10}: {}\n'.format('R2', self.metrics['r2']))
        # print('{:<10}: {}'.format('MSE', mse))
        # self.f.write('{:<10}: {}\n'.format('MSE', self.metrics['mse']))
        # print('{:<10}: {}'.format('MSE', mse))
        # self.f.write('{:<10}: {}\n'.format('RMSE', self.metrics['rmse']))
        # print('{:<10}: {}'.format('Adj_R2', adj_r2))
        # self.f.write('{:<10}: {}\n'.format('Adj_R2', self.metrics['adj_r2']))

        return self.metrics


    def display_plots(self, figsize):

        # Number of plots to display
        nrows = len(self.time_period) / 2
        
        # Create figure
        fig2 = plt.figure(2)

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

        # plt.tight_layout()
        # plt.show()
        fig2.tight_layout()
        # fig2.savefig('modeled_data.png')
        return fig2

