import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Callable

def backtester(df_forecast: pd.DataFrame, df_price: pd.DataFrame, contract_size: pd.Series=None, shift_signal: int=2, pos_mul_series: pd.Series=None, df_eligibles:pd.DataFrame=None):
    df_nominal = df_forecast.copy()
    if df_eligibles is not None:
        df_nominal = df_nominal * df_eligibles
    
    if pos_mul_series is not None:
        df_nominal = df_nominal.mul(pos_mul_series, axis='index')
    df_units = df_nominal/ (df_price * contract_size).dropna(how='all', axis=1)
    df_weights = df_nominal.div(df_nominal.abs().sum(axis=1), axis='index')
    df_nominal = df_nominal.shift(shift_signal)
    df_pnl_nominal = (df_nominal * df_price.pct_change())
    df_pnl_capital = (df_nominal.div(df_nominal.abs().sum(axis=1), axis='index') * df_price.pct_change())

    return df_weights, df_nominal, df_units, df_pnl_nominal, df_pnl_capital


def get_strat_scalar(df_pnl_nominal: pd.DataFrame, nominal_target_vol: float):
    df_pnl_rolling_vol = df_pnl_nominal.sum(axis=1).rolling(25).std().mul(np.sqrt(252)).replace(0, np.NaN)
    strat_scalar_series = nominal_target_vol/df_pnl_rolling_vol
    return strat_scalar_series


def main_backtester(df_forecast: pd.DataFrame, df_price: pd.DataFrame, contract_size: pd.Series=None,
                    shift_signal: int=2, nominal_target_vol: float=1_000_000, vol_target: bool =True,
                    shift_strat_scalar: int=2, df_eligibles: pd.DataFrame=None):
    
    if contract_size is None:
        print('No contract_size provided, contract_size is assumed to be 1')
        contract_size = pd.Series([1]*df_forecast.shape[1])

    df_weights, df_nominal, df_units, df_pnl_nominal, df_pnl_capital = backtester(df_forecast=df_forecast, df_price=df_price, contract_size=contract_size, shift_signal=shift_signal, pos_mul_series=None,
                                                                                  df_eligibles=df_eligibles)
    if vol_target:
        strat_scalar_series = get_strat_scalar(df_pnl_nominal, nominal_target_vol)
        df_weights, df_nominal, df_units, df_pnl_nominal, df_pnl_capital = backtester(df_forecast=df_forecast, df_price=df_price, contract_size=contract_size, shift_signal=shift_signal, pos_mul_series=strat_scalar_series.shift(shift_strat_scalar),
                                                                                      df_eligibles=df_eligibles)
    return df_weights, df_nominal, df_units, df_pnl_nominal, df_pnl_capital

def reindex_and_ffill(df, index, limit=10):
    dfc = df.reindex(index).ffill(limit=limit).dropna(how='all', axis=0)
    return dfc


def check_strings_and_return_missing(list1, list2):
    missing_items = [item for item in list1 if item not in list2]
    if missing_items:
        raise ValueError(f"Missing items: {missing_items}")
    return True


def reindex_and_ffill(df, index, limit=10):
    dfc = df.reindex(index).ffill(limit=limit).dropna(how='all', axis=0)
    return dfc


def get_df_eligibles(df_price, thres=5):
    df_pricec = df_price.copy().ffill()
    eligible_price_checker = (df_pricec != df_pricec.shift(1)).bfill() * 1
    df_eligibles = (eligible_price_checker.rolling(thres).mean().fillna(1) > 0)*1
    return df_eligibles

def shuffle_weights(df_position, df_eligibles, axis='time'):
    df_eligiblec = df_eligibles.reindex(df_position.index).fillna(0)
    
    
    def helper(positions, eligibles):
        new_positions = deepcopy(positions)
        for idx in range(positions.shape[1]):
            eligible_indices = np.where(eligibles[:, idx] == 1)[0]
            eligible_values = deepcopy(positions[:, idx][eligible_indices])
            np.random.shuffle(eligible_values)
            new_positions[:, idx][eligible_indices] = eligible_values
        return new_positions

    if axis == 'time':
        positions, eligibles = deepcopy(df_position.values), deepcopy(df_eligiblec.values)
        new_positions = helper(positions, eligibles)
    elif axis == 'cross':
        positions, eligibles = deepcopy(df_position.values).T, deepcopy(df_eligiblec.values).T
        new_positions = helper(positions, eligibles).T
        
    if axis == 'both':
        positions, eligibles = deepcopy(df_position.values).T, deepcopy(df_eligiblec.values).T
        new_positions = helper(positions, eligibles).T
        df_new_positions = pd.DataFrame(new_positions, index=df_position.index, columns=df_position.columns)
        positions, eligibles = deepcopy(df_new_positions.values), deepcopy(df_eligiblec.values)
        new_positions = helper(positions, eligibles)
        
    return pd.DataFrame(new_positions, index=df_position.index, columns=df_position.columns)

def execution_sensitivity_check(df_position, shift_window=1, df_eligibles=None):
    return df_position.shift(shift_window)

def calc_stats(df_pnl_capital):
    df_pnl_capitalc = df_pnl_capital.dropna(how='all', axis=0).fillna(0)
    pnl_series = df_pnl_capitalc.mean(axis=1)
    daily_ret = pnl_series.mean()
    daily_std = pnl_series.std()
    ann_ret = daily_ret * 252
    ann_std = daily_std*np.sqrt(252)
    sharpe = daily_ret/daily_std * np.sqrt(252)
    high_watermark = pnl_series.cumsum().cummax()
    average_dd = (high_watermark - pnl_series.cumsum()).mean()
    max_dd = (high_watermark - pnl_series.cumsum()).max()
    stats_dict = {'daily_ret': daily_ret,
                  'daily_std': daily_std,
                  'ann_ret': ann_ret,
                  'ann_std': ann_std,
                  'sharpe': sharpe,
                  'average_dd': average_dd,
                  'max_dd': max_dd}
    return stats_dict
