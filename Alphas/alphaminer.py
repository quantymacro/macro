import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Callable

from quantutils.backtester_utils import calc_stats
from itertools import product, combinations, permutations
import time
from multiprocessing import Process, Pool

def generate_multiindex(pre_compute_params_mega, post_compute_params_mega):
    # Merge the dictionaries and create a multiindex from all combinations
    combined_params = {**pre_compute_params_mega, **post_compute_params_mega}
    combinations = list(product(*combined_params.values()))
    multiindex = pd.MultiIndex.from_tuples(combinations, names=combined_params.keys())
    return multiindex

def generate_param_combinations_and_multi_index(pre_compute_params_mega, post_compute_params_mega):
    # Use product to generate all combinations of the parameters
    pre_compute_combinations = (dict(zip(pre_compute_params_mega, x)) for x in product(*pre_compute_params_mega.values()))
    post_compute_combinations = (dict(zip(post_compute_params_mega, x)) for x in product(*post_compute_params_mega.values()))
    
    # Combine the pre and post compute parameters
    combinations = list(product(pre_compute_combinations, post_compute_combinations))
    multiindex = generate_multiindex(pre_compute_params_mega, post_compute_params_mega)
    # Convert combinations to a list of tuples with pre and post params
    return [(pre, post) for pre, post in combinations], multiindex

def plot_parameter_scatter(df, x_level, y_level, color_column, colormap='viridis'):
    """
    Plot a scatter plot from a DataFrame with MultiIndex. Uses discrete colors for categorical data.

    Parameters:
    df (DataFrame): DataFrame with a single column and a MultiIndex.
    x_level (str): The name of the MultiIndex level to be used as x-axis.
    y_level (str): The name of the MultiIndex level to be used as y-axis.
    z_level (str): The name of the MultiIndex level to be used for color (categorical).
    
    Returns:
    A scatter plot with discrete colors.
    """
    
    # Reset index to access MultiIndex levels as columns
    df_reset = df.reset_index()
    
    # Check if z_level is categorical and use discrete colors
    if df_reset[color_column].dtype == 'object':
        categories = df_reset[color_column].unique()
        color_palette = sns.color_palette("coolwarm", len(categories))
        color_map = dict(zip(categories, color_palette))
        df_reset['colors'] = df_reset[color_column].map(color_map)

        # Create the scatter plot
        plt.figure(figsize=(10, 8))

        # Plot each category with its own color
        for category, color in color_map.items():
            subset = df_reset[df_reset[color_column] == category]
            plt.scatter(subset[x_level], subset[y_level], c=[color], label=category)

        plt.legend(title=color_column)

        # Setting the labels and title
        plt.xlabel(f'{x_level} ')
        plt.ylabel(f'{y_level} ')
        plt.title('Parameter')
        plt.show()
    else:
        plt.figure(figsize=(10, 8))
        scatter_plot = plt.scatter(x=df_reset[x_level], y=df_reset[y_level], 
                                c=df_reset[color_column], cmap=colormap)

        # Adding a color bar
        plt.colorbar(scatter_plot, label=f'{color_column}')

        # Setting the labels and title
        plt.xlabel(f'{x_level}')
        plt.ylabel(f'{y_level}')
        plt.title('Parameter')
        plt.show()


def is_column_non_numeric(df, column_name):
    return not pd.api.types.is_numeric_dtype(df[column_name])

def plot_unidimensional_parameter_curve(df_pnl_stats, stat_column):
    param_names = list(df_pnl_stats.index.names)
    df_pnl_stats_reset_index = df_pnl_stats.reset_index()
    param_types = [is_column_non_numeric(df_pnl_stats_reset_index, param) for param in param_names]
    for param_name, param_type in zip(param_names, param_types):
        stat_series = df_pnl_stats_reset_index.groupby(param_name).mean()[stat_column]
        if param_type:
            stat_series.plot.bar()
        else:
            stat_series.plot()
        plt.title(f'{stat_column} vs {param_name}')
        plt.ylabel(f'{stat_column}')
        plt.xlabel(f'{param_name}')
        plt.show()

class AlphaMiner:
    
    def __init__(self, strat, pre_compute_params_mega: dict={},
                 post_compute_params_mega: dict={}) -> None:
        
        self.strat = deepcopy(strat)
        self.pre_compute_params_mega = deepcopy(pre_compute_params_mega)
        self.post_compute_params_mega = deepcopy(post_compute_params_mega)
        
        self.combinations, self.multiindex = generate_param_combinations_and_multi_index(pre_compute_params_mega, post_compute_params_mega)
        print(f'Number Of Combinations: {len(self.combinations)}')
        
    def backtest_combination(self, combo):
        strat_copy = deepcopy(self.strat)
        strat_copy.pre_compute_params = combo[0]
        strat_copy.post_compute_params = combo[1]
        strat_copy.run_backtest()
        print(combo)
        return strat_copy
    
    def parameters_mining(self, multi=False):
        start_time = time.time()
        if multi:     
            with Pool() as p:
                res = p.map(self.backtest_combination, self.combinations)
                
        else:
            res = [self.backtest_combination(combo) for combo in self.combinations]
        self.res = res
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Time taken: {total_time} seconds")
        self.df_pnl_mega = pd.concat([strat.df_pnl_capital.sum(axis=1) for strat in res], axis=1, keys=self.multiindex)
        self.df_pnl_stats = pd.DataFrame([calc_stats(strat.df_pnl_capital) for strat in res], index=self.multiindex)
        
    def plot_parameter(self, x_axis='rolling_window', y_axis='sharpe', color_col='attribute'):
        plot_parameter_scatter(self.df_pnl_stats, x_axis, y_axis, color_col)
        plot_unidimensional_parameter_curve(self.df_pnl_stats, y_axis)
