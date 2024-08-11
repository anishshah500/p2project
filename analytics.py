import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
import plotly.graph_objs as go

from pymongo import MongoClient
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.rolling import RollingOLS
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from pykalman import KalmanFilter
from scipy.stats import zscore
from data_loader import *
from scipy.optimize import minimize
from dash import html

start_date_limit, end_date_limit = '1995-01-01', '2024-08-09'

warnings.simplefilter(action='ignore', category=FutureWarning)

class Analytics():
    """
    A class to analyze the data. 
    """
    
    def __init__(self):
        """
        Initialize the Analyzer.
        """
        self.dc = DataLoader(start_date_limit, end_date_limit)

    def set_data_df(self, start_date, end_date, force_refresh=False):
        """
        Set data and returns df
        """
        self.df = self.dc.get_equities_data(start_date, end_date, force_refresh)
        self.df.set_index("Date", inplace=True)
        self.df.index = pd.to_datetime(self.df.index).date
        
        self.returns_df = self.df / self.df.shift(1) - 1

    def generate_scatter_plot(self, index_actual, index_predicted):
        """
        Get graph for the regression
        """
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=index_actual, y=index_predicted, mode='markers', name='Actual vs fitted index'))
        
        fig.update_layout(title='Regression Analysis',
                          xaxis_title='Actual index',
                          yaxis_title='Fitted index',
                          legend_title='Legend')
        
        return fig

    def perform_regression(self, index, explain_securities):
        """
        Performs regressions of index data against the provided security symbols
        """
        filtered_df_ret = self.returns_df.loc[:, explain_securities + [index]].dropna()
        X = filtered_df_ret.loc[:, explain_securities]
        X = sm.add_constant(X)  # Add a constant for the intercept
        y = filtered_df_ret[index]

        model = sm.OLS(y, X).fit()

        filtered_df_ret[index + "_predicted"] = model.predict(X)
        
        index_actual = self.df[index].tolist()
        
        index_predicted = [index_actual[0]]
        for ret in filtered_df_ret[index + "_predicted"].tolist():
            new_price = index_predicted[-1] * (1 + ret)
            index_predicted.append(new_price)

        fig = self.generate_scatter_plot(index_actual, index_predicted)

        return model, fig
