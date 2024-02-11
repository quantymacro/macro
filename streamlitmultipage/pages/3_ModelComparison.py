import sys
sys.path.append('../') 
try:
    sys.path.remove('c:\\users\\wazir\\documents\\wazewww\\mlp')
except:
    pass
import pandas as pd
import numpy as np
import streamlit as st
import quantutils.regression_utils as regutils
import quantutils.general_utils as genutils
from quantutils.general_utils import get_df_regime_label
import quantutils.streamlit_utils as stutils
from functools import partial
from itertools import product
from tqdm.notebook import tqdm
import plotly.express as px
import plotly.graph_objs as go
from copy import deepcopy
from plotly.subplots import make_subplots
from pathlib import Path
import os
import altair as alt

### This script takes fitted models and results, and then make comparison
'''
### Things to show
- prediction over time
- scatter plot
- weighting scheme (hard to implement)
- correlation between models (both error and residuals)
- ranking of models

'''


def model_comparison(models_list, target_variable_index_name, prediction_agg_function='mean', shift=1):
    df_preds_mega = []
    df_residuals_mega = []
    sharpes = []
    df_stats_mega = []
    df_x_variables = []
    df_pnls_mega = []
    df_hit_rate_mega = []
    model_names = []
    for model in models_list:
        model_names.append(model['name'])
        df_predictions = deepcopy(model['df_coefs_dict']['predictions']).dropna()
        target_variable = model['df_transformed'][st.session_state['selected_target']]

        df_residuals = deepcopy(model['df_coefs_dict']['residuals'].dropna())
        df_stats = regutils.generate_stats_table(model).set_index('metrics')
        stats_agg = df_stats.select_dtypes('number').groupby('metrics').mean()
        pred_agg = df_predictions.mean(axis=1)
        residuals_agg = df_residuals.mean(axis=1)
        pred_agg.name = model['name']
        residuals_agg.name = model['name']
        
        
        # pnl = pred_agg.shift(shift) * target_index.pct_change()
        # pnl.name = model['name']
        # df_pnls_mega.append(pnl.cumsum())
        # sharpe = (pnl.mean()/pnl.std())
        # stats_agg.loc[len(stats_agg)] = (sharpe)
        # stats_agg.index = list(stats_agg.index[:-1]) + ['sharpe']
        
        hit_rate = np.sign(pred_agg) * np.sign(target_variable)
        # df_hit_rate_rolling = pd.concat([hit_rate.rolling(window).mean() for window in windows], axis=1)
        # df_hit_rate_rolling.columns = [f'{model["name"]}_{int(window/12)}y' for window in windows]
        
        
        stats_agg.name = model['name']
        df_preds_mega.append(pred_agg)
        df_residuals_mega.append(residuals_agg)
        df_stats_mega.append(stats_agg)
        df_hit_rate_mega.append(hit_rate)
        # sharpes.append(sharpe)
        df_x_variables.append(pd.Series(model['selected_x_variables']))
    st.write(np.sign(target_variable).value_counts())
    df_preds_mega = pd.concat(df_preds_mega, axis=1)
    df_preds_mega.index = pd.to_datetime(df_preds_mega.index)
    df_preds_mega['ensemble'] = df_preds_mega.mean(axis=1)
    df_preds_mega['naive'] = target_variable.shift(1)
    
    
    df_residuals_mega = pd.concat(df_residuals_mega, axis=1)
    df_residuals_mega['ensemble'] = ((df_preds_mega['ensemble'] - target_variable))
    df_residuals_mega['naive'] = ((df_preds_mega['naive'] - target_variable))
    
    stats_ensemble = regutils.calc_stats(df_preds_mega[['ensemble']], pd.Series(target_variable))
    stats_naive = regutils.calc_stats(df_preds_mega[['naive']], pd.Series(target_variable))
    ensemble_dict = {item[0]: item[2] for item in stats_ensemble}
    naive_dict = {item[0]: item[2] for item in stats_naive}
    df_stats_mega = pd.concat(df_stats_mega, axis=1)
    df_stats_mega.columns = [model['name'] for model in models_list]
    df_stats_mega['ensemble'] = df_stats_mega.index.map(ensemble_dict)
    df_stats_mega['naive'] = df_stats_mega.index.map(naive_dict)
    
    df_x_variables = pd.concat(df_x_variables, axis=1)
    df_x_variables.columns = [model['name'] for model in models_list]
    
    windows = [12, 36]
    df_hit_rate_mega = pd.concat(df_hit_rate_mega, axis=1)
    df_hit_rate_mega['ensemble'] = np.sign(df_preds_mega['ensemble']) * np.sign(target_variable)
    df_hit_rate_mega['naive'] = np.sign(df_preds_mega['naive']) * np.sign(target_variable)
    df_hit_rate_mega = pd.concat([df_hit_rate_mega.rolling(window).mean() for window in windows], axis=1)
    columns = list(product(model_names + ['ensemble', 'naive'], [str(int(window/12))+'y' for window in windows]))
    columns = ["_".join(col) for col in columns]
    df_hit_rate_mega.columns =  columns
    

    df_adl = np.sign(df_preds_mega.mul(model['df_transformed'][model['selected_target']], axis='index'))
    df_adl = df_adl.mul(model['df_transformed'][model['selected_target']].abs(), axis='index') * (-1)

    df_preds_mega['target'] = model['df_transformed'][model['selected_target']]

    return df_preds_mega.dropna(), df_residuals_mega.dropna(), df_stats_mega, df_x_variables, df_adl, df_hit_rate_mega

with st.form(key='Select Target Variable'):
    all_targets = os.listdir('../RegressionTools/Models')
    selected_target = st.selectbox('Target Variable', all_targets)
    target_variable_index_name = selected_target.split('_')[0]
    st.session_state['selected_target'] = selected_target
    st.session_state['target_variable_index_name'] = target_variable_index_name
    submit_button_selected_target = st.form_submit_button(label='Confirm Target Variable')

if submit_button_selected_target:
    models_list = []
    directory = f'../RegressionTools/Models/{st.session_state["selected_target"]}/'
    model_files = os.listdir(directory)
    for file_name in model_files:
        model = genutils.load_object(file_name, directory)
        models_list.append(model)

    st.session_state['models_list'] = models_list

if 'models_list' in st.session_state:
    df_preds_mega, df_residuals_mega, df_stats_mega, df_x_variables, df_adl, df_hit_rate_mega = model_comparison(st.session_state['models_list'], st.session_state['target_variable_index_name'])
    st.write('#### Selected X Variables')
    st.write(df_x_variables)
    st.write('#### Stats Comparison')
    st.dataframe(df_stats_mega)
    # st.write('#### Naive PnL')
    # st.line_chart(df_pnls_mega.dropna())

    st.write('#### Time Series of Predictions vs Target')
    st.line_chart(df_preds_mega)
    
    st.write('#### Hit Rate 1y and 3y')
    
    df_reset = df_hit_rate_mega.dropna().reset_index().rename(columns={'index': 'Date'})

    # Melt the DataFrame so that it's in long form for Altair
    df_melted = df_reset.melt(id_vars=df_reset.columns[0], 
                            value_vars=df_reset.columns[1:], 
                            var_name='Series', value_name='HitRate')

    # Create the line chart using Altair
    chart = alt.Chart(df_melted).mark_line().encode(
        x=alt.X(df_reset.columns[0], type='temporal'),  # Assuming the first column after reset is the date
        y=alt.Y('HitRate:Q'),
        color='Series:N',
        tooltip=[df_reset.columns[0], 'Series', 'HitRate']
    ).properties(
        width=900,  # Adjust the width as necessary
        height=300  # Adjust the height as necessary
    ).interactive()

    # Display the chart in Streamlit
    st.altair_chart(chart)
        
    

    residual_corr = deepcopy((df_residuals_mega)).corr()
    predictions_corr = deepcopy((df_preds_mega.iloc[:, :-1])).corr()
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Prediction Correlation', 'Residuals Correlation'), horizontal_spacing=0.2)
    fig.add_trace(
        go.Heatmap(
            z=residual_corr.values,
            x=residual_corr.columns.tolist()[::-1],
            y=residual_corr.index.tolist(),
            colorscale='RdBu',
            showscale=True,
            zmid=0, # Center the color scale at zero
            text=residual_corr.values.round(3),  # Add the correlation values as text
            texttemplate="%{text}",  # Use the text from 'text' as the template
            textfont={"size":15, "color":"white"},
        ),
        row=1, col=2
    )

    # Add the second heatmap to the second column
    fig.add_trace(
        go.Heatmap(
            z=predictions_corr.values,
            x=predictions_corr.columns.tolist()[::-1],
            y=predictions_corr.index.tolist(),
            colorscale='RdBu',
            showscale=True,
            zmid=0,  # Center the color scale at zero
            text=predictions_corr.values.round(3),  # Add the correlation values as text
            texttemplate="%{text}",  # Use the text from 'text' as the template
            textfont={"size":15, "color":"white"},
        ),
        row=1, col=1
    )

    # Update the layout
    fig.update_layout(
        title_text='Correlation of Prediction and Residuals',
        showlegend=False,
    )

    st.plotly_chart(fig)

    df_ranks = df_residuals_mega.abs().rolling(12).mean().rank(axis=1, ascending=True)
    fig = stutils.plotly_ranking(df_ranks, 'Model Ranking Over Time')
    st.plotly_chart(fig)

    df_residual = deepcopy((df_residuals_mega).abs()**2).rolling(30).mean().dropna()

    ##Rolling MSE of models
    st.write('#### Rolling MSE')
    st.line_chart(df_residual)

    st.line_chart(df_adl.rolling(12).mean())
