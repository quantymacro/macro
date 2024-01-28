import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Callable
from Alphas.mainalpha import Alpha
from quantutils.backtester_utils import *


class FXTrendTS(Alpha):
    
    def __init__(self, df_dict: dict, start: str, end: str,
                 nominal_target_vol: float, contract_size: dict, shift_signal: int=2,
                 pre_compute_params: dict={}, post_compute_params: dict={}):
        super().__init__(df_dict, start, end,
                         nominal_target_vol, contract_size, shift_signal,
                         pre_compute_params, post_compute_params)
        
    def pre_compute(self, date_range: pd.DatetimeIndex=None):
        if date_range is None:
            date_range = self.date_range
        pre_compute_compulsory_params = ['window_pair']
        check_strings_and_return_missing(pre_compute_compulsory_params, list(self.pre_compute_params.keys()))

        fast_window, slow_window = self.pre_compute_params['window_pair']
        df_ma_fast = self.df_price.rolling(fast_window).mean()
        df_ma_slow = self.df_price.rolling(slow_window).mean()
        df_price_diff = self.df_price.diff().rolling(30).std()
        df_alpha = (df_ma_fast - df_ma_slow)/df_price_diff
        self.df_alpha = df_alpha.dropna(how='all', axis=0)
        
    def post_compute(self, date_range: pd.DatetimeIndex=None):
        if date_range is None:
            date_range = self.date_range
        self.df_forecast = self.df_alpha
    
    def compute_signal_distribution(self):
        pass