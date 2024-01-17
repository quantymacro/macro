import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Callable
from quantutils.backtester_utils import *
from Alphas.mainalpha import Alpha

class Strat2(Alpha):
    
    def __init__(self, df_dict: dict, start: str, end: str,
                 nominal_target_vol: float, contract_size: dict, shift_signal: int=2,
                 pre_compute_params: dict={}, post_compute_params: dict={}):
        super().__init__(df_dict, start, end,
                         nominal_target_vol, contract_size, shift_signal,
                         pre_compute_params, post_compute_params)
        
    def pre_compute(self, date_range: pd.DatetimeIndex):
        if date_range is None:
            date_range = self.date_range
            
        df_trend = deepcopy(self.df_dict['trend'])
        df_forecast = reindex_and_ffill(df_trend, date_range)
        self.df_forecast = df_forecast.dropna(how='all', axis=0)
    
    def post_compute(self, date_range: pd.DatetimeIndex=None):
        if date_range is None:
            date_range = self.date_range
        self.df_forecast = self.df_alpha
    
    def compute_signal_distribution(self):
        pass  
        