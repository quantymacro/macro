import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Callable
from Alphas.mainalpha import Alpha
from quantutils.backtester_utils import *


class FXTrend(Alpha):
    
    def __init__(self, df_dict: dict, start: str, end: str,
                 nominal_target_vol: float, contract_size: dict, shift_signal: int=2,
                 pre_compute_params: dict={}, post_compute_params: dict={}):
        super().__init__(df_dict, start, end,
                         nominal_target_vol, contract_size, shift_signal,
                         pre_compute_params, post_compute_params)
        
    def pre_compute(self, date_range: pd.DatetimeIndex=None):
        ## compute df_alpha
        if date_range is None:
            date_range = self.date_range
            
        df_fast_ma = self.df_price.rolling(10).mean()
        df_slow_ma = self.df_price.rolling(40).mean()
        df_alpha = df_fast_ma - df_slow_ma
        df_alpha = df_alpha/df_alpha.rolling(30).std()
        self.df_alpha = df_alpha.dropna(how='all', axis=0)
        
    def post_compute(self, date_range: pd.DatetimeIndex=None):
        ## df_forecast
        if date_range is None:
            date_range = self.date_range
        self.df_forecast = self.df_alpha
    
    def compute_signal_distribution(self):
        pass