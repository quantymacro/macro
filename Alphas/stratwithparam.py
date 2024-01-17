import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Callable
from quantutils.backtester_utils import *
from Alphas.mainalpha import Alpha

class StratWithParam(Alpha):
    def __init__(self, df_dict: dict, start: str, end: str,
                 nominal_target_vol: float, contract_size: dict, shift_signal: int=2,
                 pre_compute_params: dict={}, post_compute_params: dict={}):
        
        super().__init__(df_dict, start, end,
                         nominal_target_vol, contract_size, shift_signal,
                         pre_compute_params, post_compute_params)

        
    def pre_compute(self, date_range: pd.DatetimeIndex=None):
        
        pre_compute_compulsory_params = ['attribute']        
        check_strings_and_return_missing(pre_compute_compulsory_params, list(self.pre_compute_params.keys()))
        
        if date_range is None:
            date_range = self.date_range
            
        attribute = self.pre_compute_params['attribute']
        df_alpha = reindex_and_ffill(self.df_dict[attribute], date_range)
        self.df_alpha = df_alpha.dropna(how='all', axis=0)
    
    def post_compute(self, date_range: pd.DatetimeIndex=None):
        
        post_compute_compulsory_params = ['rolling_window', 'mult']
        check_strings_and_return_missing(post_compute_compulsory_params, list(self.post_compute_params.keys()))

        rolling_window = self.post_compute_params['rolling_window']
        mult = self.post_compute_params['mult']
        if date_range is None:
            date_range = self.date_range
        self.df_forecast = self.df_alpha.rolling(rolling_window).mean() + np.random.normal(size=self.df_alpha.shape, scale=mult)
        
    def compute_signal_distribution(self):
        pass
    