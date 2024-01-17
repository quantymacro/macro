import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Callable

# Now you can import Strat1
from Alphas.strat1 import Strat1
from Alphas.strat2 import Strat2
from Alphas.stratwithparam import StratWithParam
from quantutils.backtester_utils import Alpha
from itertools import product, combinations, permutations
from Alphas.alphaminer import AlphaMiner
from multiprocessing import Process, Pool
import time

np.random.seed(22)
contract_size = pd.Series([25, 1, 5, 3, 10])
T = 5000*2
N = 10
std_mult = np.random.rand(1, 5, N)
df_ret = pd.DataFrame(np.random.normal(loc=0.001, scale=0.02, size=(T, 5)))
df_ret.index = pd.date_range('2010-01-01', periods=T)
df_ret = df_ret *std_mult+ 1
df_ret.iloc[0] = np.random.randint(1, 101, N)
df_price =  (df_ret).cumprod()
df_ret = df_price.pct_change()
attributes = ['high', 'close', 'low', 'max', 'trend']
df_dict = {}
for attribute in attributes:
    window = np.random.randint(5, 101)
    df_signal = (df_price.pct_change(window).shift(-window+1) + np.random.normal(scale=0.5, size=df_ret.shape)).shift().ffill().bfill() * 100
    df_signal = df_signal.rolling(20).mean()

    df_signal = df_signal/df_signal.std()
    df_dict[attribute] = df_signal
    
df_dict['price'] = df_price
df_dict['low'] = df_dict['low'].drop(0 ,axis=1)
contract_size = {k : np.random.randint(1, 20) for k in range(df_ret.shape[1])}


def main():
    pre_compute_params_mega = {'attribute': ['high', 'low']}
    post_compute_params_mega = {'rolling_window': np.random.randint(5, 1000, 600), 'mult': [0.1, 0.2]}    
    start = '2010-01-01'
    end = '2023-01-01'
    strat_skeleton = StratWithParam(df_dict, start, end, 1_000_000, contract_size, 2)
    alphaminer = AlphaMiner(strat_skeleton, pre_compute_params_mega, post_compute_params_mega)
    alphaminer.parameters_mining(True)
    
if __name__ == '__main__':
    main()
    
    