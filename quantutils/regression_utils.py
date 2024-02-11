import numpy as np
import pandas as pd
import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, Ridge
from scipy.stats import zscore, mstats
from constrained_linear_regression import ConstrainedLinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from statsmodels.tsa import x13


def rolling_regression_sklearn_advanced(data, rolling_window, n_step_ahead=1, 
                                        l1_ratio=0.1, 
                                        dropna=False, remove_outliers=False, 
                                        winsorize=False, winsorize_limits=(0.05, 0.95),
                                        fit_intercept=False, min_coef=None, max_coef=None,
                                        expanding=False):
    """
    Perform rolling regression from sklearn with additional data processing options.
    
    Parameters:
        data (pd.DataFrame): DataFrame where one of the columns should be "target". 
                             Should have a DateTimeIndex.
        rolling_window (int): Number of samples to use for each regression.
        n_step_ahead (int, optional): Number of steps ahead to predict. Default is 1.
        l1_ratio (float, optional): The L1 regularization ratio. Default is 0.1.
        dropna (bool, optional): Whether to drop NaN values. Default is False.
        remove_outliers (bool, optional): Whether to remove outliers based on Z-score. Default is False.
        winsorize (bool, optional): Whether to winsorize data. Default is False.
        winsorize_limits (tuple, optional): Percentiles for winsorizing. Default is (0.05, 0.95).
    
    Returns:
        pd.DataFrame: Coefficients for each window.
        pd.Series: Predictions.
    """
    # Drop NaN values if requested
    datac = data.copy()
    if dropna:
        datac = datac.dropna()
    n_samples, n_features_plus_one = datac.shape
    n_features = n_features_plus_one - 1
    coefs = pd.DataFrame(index=datac.index, columns=datac.drop('target', axis=1).columns)
    predictions = pd.Series(index=datac.index, name='predictions')
    X_series = pd.Series(index=datac.index, name='x', dtype='object')
    y_series = pd.Series(index=datac.index, name='y', dtype='object')
    fitted_models = pd.Series(index=datac.index, name='model', dtype='object')
    for start in range(0, n_samples - rolling_window - n_step_ahead + 1, n_step_ahead):
        if expanding:
            window = datac.iloc[0:start + rolling_window].copy()  # Use copy to avoid SettingWithCopyWarning
        else:
            window = datac.iloc[start:start + rolling_window].copy()  # Use copy to avoid SettingWithCopyWarning
        
        # Remove outliers if requested
        if remove_outliers:
            z_scores = np.abs(zscore(window))
            window = window[(z_scores < 3).all(axis=1)]
        
        # Winsorize data if requested
        if winsorize:
            window = window.apply(lambda col: mstats.winsorize(col, limits=winsorize_limits), axis=0)
        
        X, y = window.drop('target', axis=1), window['target']

        model = ConstrainedLinearRegression(ridge=l1_ratio, normalize=True, fit_intercept=fit_intercept)
        model.fit(X, y, min_coef=min_coef, max_coef=max_coef)

        end_idx = start + rolling_window
        ratio = y.std()/model.predict(X).std()
        model.coef_ = model.coef_ * ratio
        X_series.iloc[end_idx] = X
        y_series.iloc[end_idx] = y
        coefs.iloc[end_idx] = model.coef_
        future_X = datac.iloc[end_idx:end_idx + n_step_ahead, :-1]
        future_preds = model.predict(future_X) 
        predictions.iloc[end_idx:end_idx + n_step_ahead] = future_preds
        fitted_models.iloc[end_idx] = model
    df_results = pd.concat([coefs.ffill(limit=30), predictions], axis=1).reindex(data.index).ffill(limit=10)
    return df_results, fitted_models, X_series, y_series


def create_heatmap_and_table(df, columns, y_col_name):
    effective_columns = [col for col in columns if 'l0' not in col] + [y_col_name]
    corr = df[effective_columns].corr()
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(corr, annot=False)
    # plt.tight_layout()
    # plt.savefig('heatmap.png', bbox_inches='tight')
    # plt.close()
    
    rounded_data = corr[y_col_name].round(3).sort_values(ascending=False)
    cmap = sns.color_palette("icefire", as_cmap=True)
    norm = Normalize(vmin=rounded_data.min(), vmax=rounded_data.max())

    # Adjusting the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a table with 'icefire' colormap
    table = plt.table(cellText=rounded_data.values[:, np.newaxis], 
                    rowLabels=rounded_data.index, 
                    colLabels=['Value'], 
                    cellColours=cmap(norm(rounded_data.values[:, np.newaxis])),
                    loc='center',
                    cellLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(rounded_data)))) 

    # Hide axes
    ax.axis('off')

    # Adding a colorbar with 'icefire' colormap
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.05)
    plt.tight_layout()
    plt.savefig('corrtable.png', bbox_inches='tight')
    plt.close()
    
def seasonal_adjustment(series, PATH=None):
    if PATH is None:
        PATH = 'C:/Users/Wazir/Documents/wazewww/MLP/ARIMA/winx13ascii_v3-1/WinX13/x13as'

    res = x13.x13_arima_analysis(series, x12path=PATH)
    return res

def bulk_feature_engineering(df, reg_variables_dict, y_variable):
    '''
    y-variable transformation
    - lags 1-6 and lag 12
    - 2m MA lag 1 to lag 3
    - 3m MA lag 1 to lag 3
    
    x-variable transformation
    contemporaneous
    -lags 1-4
    - 2m MA lag 1 to lag 3
    - 3m MA lag 1 to lag 3
    
    The convention is as follow:
    COLNAME_SEASONAL_TRANSFORMATION_LAG_(FUNC)
    '''
    x_lags = [i for i in range(0, 5)]
    dfc = pd.DataFrame(index=df.index)
    seasonal_adj_dict = {}
    for col in reg_variables_dict.keys():
        
        transformation, seasonal = reg_variables_dict[col].transformation, reg_variables_dict[col].seasonal
        seasonal_str = 'sa' if seasonal else 'nsa'
        if seasonal:
            print(f'seasonally adjusting {col}...')
            series = df[col].dropna().copy()
            res = seasonal_adjustment(series)
            series = res.seasadj
            seasonal_adj_dict[col] = res

        if transformation == 'lvl':
            series = series.copy()
            
        elif transformation == 'pct_change':
            series = series.pct_change()
        
        elif transformation == 'diff':
            series = series.diff()
            
        
        
        for x_lag in x_lags:
                
            dfc[f'{col}_{seasonal_str}_{transformation}_l{x_lag}_ma2'] = series.shift(x_lag).rolling(2).mean()
            dfc[f'{col}_{seasonal_str}_{transformation}_l{x_lag}_ma3'] = series.shift(x_lag).rolling(3).mean()

        if col == y_variable:
            for lag in x_lags[1:]:
                series = df[col].copy()
                if transformation == 'pct_change':
                    dfc[f'{col}_{seasonal_str}_{transformation}{lag}_l0'] = series.pct_change(lag)
                
                elif transformation == 'diff':
                    dfc[f'{col}_{seasonal_str}_{transformation}{lag}_l0'] = series.diff(lag)
        
    return dfc, seasonal_adj_dict


def feature_engineering(df, transformations):
    '''
    Extends the dataframe with engineered features based on the specified transformations.
    
    Parameters:
        df (pd.DataFrame): The original dataframe.
        transformations (dict): A dictionary with keys as column names and values as lists of transformations.
    
    Returns:
        pd.DataFrame: The dataframe with added engineered features.
    '''
    engineered_df = df.copy()
    
    for column, transformation_list in transformations.items():
        if column in engineered_df.columns:
            for transformation in transformation_list:
                operation, window = transformation  # Unpack operation and parameters
                if operation in ['mean', 'std', 'kurt', 'pct_change', 'diff']:
                    if operation in ['mean', 'std', 'kurt']:
                        transformed_series = getattr(df[column].rolling(window=window), operation)()
                    elif operation == 'pct_change':
                        transformed_series = df[column].pct_change(periods=window)
                    elif operation == 'diff':
                        transformed_series = df[column].diff(periods=window)
                    engineered_df[f"{column}_{operation}_{window}period"] = transformed_series
    return engineered_df

def lag_variables(df, lag_dict):
    dfc = df.copy()
    for col, windows in lag_dict.items():
        if col in dfc.columns:
            for window in windows:
                dfc[f'{col}_lag{window}'] = dfc[col].shift(window)

    return dfc




def natural_lag(df):

    cols_not_lagged = [col for col in df.columns if 'lag' not in col]
    dfc = df.copy()
    for col in cols_not_lagged:
        dfc[f'{col}_natlag1'] = dfc[col].shift(1)

    return dfc

def calc_stats(y_pred, y_true):
    
    y_truec = y_true.reindex(y_pred.index).dropna()
    y_predc = y_pred.reindex(y_truec.index)

    stats = [('r2', col, r2_score(y_truec.loc[y_predc.index].values, y_predc[col])) for col in y_pred.columns]  + \
        [('mse', col, mean_squared_error(y_truec.loc[y_predc.index].values, y_predc[col])) for col in y_pred.columns] + \
        [('mae', col, mean_absolute_error(y_truec.loc[y_predc.index].values, y_predc[col])) for col in y_pred.columns] + \
        [('corr', col, np.corrcoef(y_truec.loc[y_predc.index].values, y_predc[col].values)[0][1]) for col in y_pred.columns] + \
        [('hit_rate', col, (np.sign(y_truec.loc[y_predc.index].values) * np.sign(y_predc[col].values)).mean()) for col in y_pred.columns]
        
    return stats

def generate_stats_table(regression_streamlit_state, y_pred=None, y_true=None):
    y_pred = regression_streamlit_state['df_coefs_dict']['predictions'].dropna()
    y_true = regression_streamlit_state['df_transformed'][regression_streamlit_state['selected_target']].loc[y_pred.index].dropna()
    y_pred = y_pred.loc[y_true.index]
    
    stats = calc_stats(y_pred,y_true)
    
    df_stats = pd.DataFrame(stats, columns=['metrics', 'model', 'value'])
    return df_stats

def mean_absolute_directional_loss(y_true, y_pred):
    adl = np.sign(y_true * y_pred) * np.abs(y_true) * (-1)
    madl = np.mean(adl)
    return madl