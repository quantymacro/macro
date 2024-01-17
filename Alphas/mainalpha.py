import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Callable
from quantutils.backtester_utils import *
from tqdm.notebook import tqdm

class AbstractImplementationException(Exception):
    pass

class Alpha():

    def __init__(self, df_dict: dict, start: str, end: str,
                 nominal_target_vol: float, contract_size: dict, shift_signal: int=2,
                 pre_compute_params: dict={}, post_compute_params: dict={}):
        self.df_dict = deepcopy(df_dict)
        self.start = start 
        self.end = end
        self.nominal_target_vol = nominal_target_vol
        self.df_price = df_dict['price']
        self.contract_size = pd.Series(contract_size)
        self.shift_signal = shift_signal
        self.pre_compute_params = pre_compute_params
        self.post_compute_params = post_compute_params
        self.date_range = pd.date_range(start=self.start,end=self.end, freq="D")
        
    def pre_compute(self, date_range: pd.DatetimeIndex=None):
        '''
        Compute all the things you need for the signal
        '''
        pass

    def post_compute(self, date_range: pd.DatetimeIndex=None):
        '''
        Compute df_forecast
        '''
        pass

    def compute_signal_distribution(self):
        raise AbstractImplementationException("no concrete implementation for signal generation")

    def compute_meta_info(self):
        self.df_eligibles = get_df_eligibles(self.df_price, 5)
    
    def run_backtest(self):
        if not hasattr(self, 'df_eligibles'):
            self.compute_meta_info() ##get_df_eligibles
        if not hasattr(self, 'df_alpha'):
            self.pre_compute(self.date_range) ##get df_forecast
        if not hasattr(self, 'df_forecast'):
            self.post_compute()
        if not hasattr(self, 'contract_size'):
            self.contract_size = None
        if not hasattr(self, 'df_eligibles'):
            self.df_eligibles = None
            
        self.compute_signal_distribution()

        df_weights, df_nominal, df_units, df_pnl_nominal, df_pnl_capital = main_backtester(df_forecast=self.df_forecast, df_price=self.df_price, contract_size=self.contract_size, shift_signal=self.shift_signal,
                                                                                           nominal_target_vol=self.nominal_target_vol, vol_target=True, shift_strat_scalar=2,
                                                                                           df_eligibles=self.df_eligibles)
        self.df_weights = df_weights.dropna(how='all', axis=0)
        self.df_nominal = df_nominal.dropna(how='all', axis=0)
        self.df_units = df_units.dropna(how='all', axis=0)
        self.df_pnl_nominal = df_pnl_nominal.dropna(how='all', axis=0)
        self.df_pnl_capital = df_pnl_capital.dropna(how='all', axis=0)

    def run_robustness_tests(self, n_sim=10):
        '''
        Generates df_robustness_tests
        '''
        def helper(self, generator_func: Callable, kwargs:dict):
            selfc = deepcopy(self)
            selfc.pre_compute()
            selfc.post_compute()
            selfc.df_forecast = generator_func(df_position=selfc.df_forecast, df_eligibles=self.df_eligibles, **kwargs)

            selfc.run_backtest()
            stats = calc_stats(selfc.df_pnl_capital)
            return stats
        
        cross_section_stats_list = [helper(self, shuffle_weights, {'axis':'cross'}) for _ in tqdm(range(n_sim))]
        time_stats_list = [helper(self, shuffle_weights, {'axis':'time'})  for _ in tqdm(range(n_sim))]
        shuffle_both_axis_list = [helper(self, shuffle_weights, {'axis':'both'})  for _ in tqdm(range(n_sim))]
        execution_sensitivity_checks = [helper(self, execution_sensitivity_check, {'shift_window': i}) for i in range(n_sim)]
        self.df_robustness_stats = pd.concat([pd.DataFrame(cross_section_stats_list),
                                              pd.DataFrame(time_stats_list),
                                              pd.DataFrame(shuffle_both_axis_list),
                                              pd.DataFrame(execution_sensitivity_checks)], axis=1, keys=['Cross-Section', 'Time-Series', 'Both', 'Execution'])
        
    
    def pnl_decomposition(self, groupings: dict=None, cost_list: list = [0.01, 0.001], df_factors: pd.DataFrame = None, plot: bool=False):
        ### Long-Short-Decomposition
        if not hasattr(self, 'df_pnl_capital'):
            self.run_backtest()
        
        def get_pnl_with_tc(self, tc: float):
            selfc = deepcopy(self)
            df_nominal_tc = selfc.df_nominal.diff().abs() * tc
            selfc.df_pnl_nominal = selfc.df_pnl_nominal - df_nominal_tc
            selfc.df_pnl_capital = selfc.df_pnl_nominal.div(selfc.df_nominal.abs().sum(axis=1), axis='index').fillna(0)
            return selfc
        
        ### TC Decomposition
        strat_with_tc = [(get_pnl_with_tc(self, tc)) for tc in cost_list]
        stats_with_tc = [calc_stats(self_.df_pnl_capital) for self_ in strat_with_tc]
        
        df_stats_with_tc = pd.DataFrame(stats_with_tc, index=cost_list)    
        self.df_pnl_with_tc = pd.concat([(self_.df_pnl_capital) for self_ in strat_with_tc], axis=1, keys=cost_list)
        self.df_stats_with_tc = df_stats_with_tc
        
        ### Long-Short Decomposition
        df_pnl_capital_long = self.df_pnl_capital * (np.sign(self.df_forecast) == 1)
        df_pnl_capital_short = self.df_pnl_capital * (np.sign(self.df_forecast) == -1)
        self.df_pnl_capital_long_short = pd.concat([df_pnl_capital_long, df_pnl_capital_short], axis=1, keys=['Long', 'Short']).fillna(0)

        ### Factor Decomposition
        
        ### Group Decomposition
        if groupings is not None:
            groupings_relevant = {key: [item for item in value if (item) in self.df_pnl_capital.columns] for key, value in groupings.items()}
            self.df_pnl_by_group = pd.concat([self.df_pnl_capital[group_cols] for group_cols in list(groupings_relevant.values())], axis=1, keys=list(groupings_relevant.keys()))
            self.df_weights_by_group = pd.concat([self.df_weights[group_cols] for group_cols in list(groupings_relevant.values())], axis=1, keys=list(groupings_relevant.keys()))