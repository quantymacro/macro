import sys
sys.path.append('../') 
try:
    sys.path.remove('c:\\users\\wazir\\documents\\wazewww\\mlp')
except:
    pass

import streamlit as st
import pandas as pd
import numpy as np
import quantutils.regression_utils as regutils
import quantutils.general_utils as genutils
from quantutils.general_utils import get_df_regime_label
import quantutils.streamlit_utils as stutils
import os
import datetime
from copy import deepcopy
from tqdm.notebook import tqdm

class Variable:
    def __init__(self, transformation=None, seasonal=False, y_variable=False):
        self.transformation = transformation
        self.seasonal = seasonal
        self.y_variable = y_variable

# Generate a sample DataFrame
def create_sample_data():
    np.random.seed(0)
    data = np.random.rand(100, 5)  # 100 rows and 5 columns
    columns = ['A', 'B', 'C', 'D', 'E']
    index = pd.date_range('2000-01-01', periods=100)
    return pd.DataFrame(data, columns=columns, index=index)

df = create_sample_data()

reg_variables_dict = {}

# Streamlit app starts here
st.title('Regression Tool')

with st.form(key='regression_form'):
    st.write('Select Y-variable')
    col1, col2, col3 = st.columns(3)

    with col1:
        y_col_name = st.selectbox('Y Column', options=df.columns, key='y_col_name')

    with col2:
        y_transformation = st.selectbox('Y Transformation', options=['lvl', 'pct_change', 'diff'], key='y_transformation')

    with col3:
        y_seasonal = st.checkbox('Y Seasonally Adjusted', key='y_seasonal')

    reg_variables_dict[y_col_name] = Variable(y_transformation, y_seasonal, True)
    st.session_state['y-variable'] = y_col_name
    st.write('Select X-variables')
    x_col_names = st.multiselect('X Columns', options=df.columns, key='x_col_names')

    # Initialize dictionaries to store the transformation and seasonal widgets for the X-variables
    x_transformations = {}
    x_seasonals = {}
    
    # Create columns and widgets for each selected X-variable
    for x_col in x_col_names:
        with st.container():  # Use a container for better layout control
            col_transformation, col_seasonal = st.columns(2)  # Create two columns
            with col_transformation:
                x_transformations[x_col] = st.selectbox(f'{x_col} Transformation', options=['lvl', 'pct_change', 'diff'], key=f'{x_col}_transformation')
            with col_seasonal:
                x_seasonals[x_col] = st.checkbox('Seasonally Adjusted', key=f'{x_col}_seasonal')

            # After creating the widgets, store their settings in the reg_variables_dict
            reg_variables_dict[x_col] = Variable(x_transformations[x_col], x_seasonals[x_col])

    submit_button = st.form_submit_button('Submit')

if submit_button:
    df_feature = regutils.bulk_feature_engineering(df, reg_variables_dict)
    image_path = "heatmap.png"
    
    regutils.create_heatmap(df, x_col_names, image_path)
    
    st.session_state['df_transformed'] = df_feature    
    st.write(df_feature.head())


if 'df_transformed' in st.session_state:
    with st.form('Variables in regression'):
        st.write('### Confirm Y-Variable')
        selected_target = st.selectbox('Y-variable', st.session_state['df_transformed'].columns)
        st.write('### Confirm X-Variables')
        selected_x_variables = st.multiselect('X-variables', st.session_state['df_transformed'].columns)
        st.session_state['selected_x_variables'] = selected_x_variables
        st.session_state['selected_target'] = selected_target
        submit_x_variable_button = st.form_submit_button(label='Confirm Variables in Regression')
        
    
if 'selected_x_variables' in st.session_state:
    # Initialize dictionaries if not already present
    # if 'min_coefs_dict' not in st.session_state:
    st.session_state['min_coefs_dict'] = {xvar: -np.inf for xvar in st.session_state['selected_x_variables']}
    # if 'max_coefs_dict' not in st.session_state:
    st.session_state['max_coefs_dict'] = {xvar: np.inf for xvar in st.session_state['selected_x_variables']}
    col1, col2 = st.columns(2)
    col1.header("Minimum Coefficients")
    col2.header("Maximum Coefficients")
    # Create a Nx2 panel for min and max coefficient constraints
    for xvar in st.session_state['selected_x_variables']:
        with col1:
            st.session_state['min_coefs_dict'][xvar] = st.number_input(f"Min coef for {xvar}", value=float(-999999), format="%f", key=f"min_{xvar}")
        with col2:
            st.session_state['max_coefs_dict'][xvar] = st.number_input(f"Max coef for {xvar}", value=float(9999999), format="%f", key=f"max_{xvar}")
    
    # Button to confirm the constraints
    if st.button('Confirm Coefficient Constraints'):
        st.write("Min Coefficients Constraints:", st.session_state['min_coefs_dict'])
        st.write("Max Coefficients Constraints:", st.session_state['max_coefs_dict'])

with st.form('Rolling Windows'):
    numbers_str = st.text_input('Enter a list of numbers, separated by commas')
    # Initialize an empty list to hold integers
    numbers_list = []
    # Convert the string to a list of integers
    if numbers_str:
        try:
            # Split the string into a list where each number is a string
            numbers_str_list = numbers_str.split(',')
            # Convert each string number to an integer and add to the list
            numbers_list = [int(number.strip()) for number in numbers_str_list]
            st.success('List of windows: ' + str(numbers_list))
        except ValueError:
            # If conversion fails, inform the user
            st.error('Please enter a valid list of numbers, separated by commas')
    submit_windows_button = st.form_submit_button(label='Confirm selected windows')
    st.session_state['windows'] = numbers_list
    
with st.form('Refit Model Frequency'):
    n_step_ahead = st.number_input('Refitting Model Frequency (1 means daily fit)', value=1)
    submit_refit_freq = st.form_submit_button(label='Confirm Re-fit Model Freq')
    expanding = st.checkbox('Expanding Window')
    if expanding:
        st.session_state['expanding'] = True
        print('Expanding Window Mode Enabled')
    else:
        st.session_state['expanding'] = False
    st.session_state['n_step_ahead'] = n_step_ahead
    
with st.form('Start Date'):
    if 'df_transformed' in st.session_state:
        start_date = st.date_input('Start Date', value=datetime.date(2000, 1, 1), format='YYYY/MM/DD', max_value=datetime.date.today())
        submit_refit_freq = st.form_submit_button(label='Confirm Start Date')
        st.session_state['start_date'] = pd.Timestamp(start_date)
        st.session_state['df_transformed'] = st.session_state['df_transformed'][start_date:]
        
if 'df_transformed' in st.session_state and 'selected_x_variables' and 'n_step_ahead' and 'start_date' in st.session_state:
    if st.button('Run Regression'):
            df_regression = st.session_state['df_transformed'][st.session_state['selected_x_variables']].copy()
            df_regression['target'] = st.session_state['df_transformed'][st.session_state['selected_target']]
            # df_regression = df_regression[st.session_state['start_date']:].copy()
            df_coefs_dict = {k: [] for k in st.session_state['selected_x_variables']}
            df_coefs_dict['predictions'] = []
            min_coef = []
            max_coef = []
            for xvar in df_regression.drop('target', axis=1).columns:
                min_coef.append(st.session_state['min_coefs_dict'][xvar])
                max_coef.append(st.session_state['max_coefs_dict'][xvar])

            st.session_state['X_dataframes'] = []
            st.session_state['y_dataframes'] = []
            st.session_state['fitted_models'] = []
            for window in tqdm(st.session_state['windows']):
                df_results, fitted_models, X_series, y_series = regutils.rolling_regression_sklearn_advanced(df_regression, rolling_window=window, n_step_ahead=st.session_state['n_step_ahead'],
                                                                            dropna=True, min_coef=min_coef, max_coef=max_coef, expanding=st.session_state['expanding'])
                
                for x_var in st.session_state['selected_x_variables']:
                    df_coefs_dict[x_var].append(df_results[[x_var]].rename(columns={f'{x_var}': f'{window}'}).squeeze())
                df_coefs_dict['predictions'].append(df_results[['predictions']].rename(columns={f'predictions': f'predictions_{window}'}).squeeze())

                st.session_state['X_dataframes'].append(X_series.rename(f'{X_series.name}_{window}'))
                st.session_state['y_dataframes'].append(y_series.rename(f'{y_series.name}_{window}'))
                st.session_state['fitted_models'].append(fitted_models.rename(f'{fitted_models.name}_window'))
            st.session_state['X_dataframes'] = pd.concat(st.session_state['X_dataframes'], axis=1)
            st.session_state['y_dataframes'] = pd.concat(st.session_state['y_dataframes'], axis=1)
            st.session_state['fitted_models'] = pd.concat(st.session_state['fitted_models'], axis=1)

            for x_var in st.session_state['selected_x_variables']:
                df_coefs_dict[x_var] = pd.concat(df_coefs_dict[x_var], axis=1).dropna()

            df_coefs_dict['predictions'] = pd.concat(df_coefs_dict['predictions'], axis=1).dropna()
            df_coefs_dict['residuals'] = df_coefs_dict['predictions'].sub(df_regression['target'], axis='index').dropna()
            df_coefs_dict['residuals'].columns = [f'residuals_{str(col).split("_")[-1]}' for col in df_coefs_dict['residuals'].columns]
            st.session_state['df_coefs_dict'] = df_coefs_dict

            regression_streamlit_state = deepcopy(dict(st.session_state))
            stutils.regression_analytics(regression_streamlit_state, df_regime_labelled=None)
    else:
        st.write('Cannot run regression')
        
        
        
with st.form(key='save_model_form'):
    name = st.text_input('Model Name')
    st.session_state['name'] = name
    save_model_submit = st.form_submit_button('Save Model')

if save_model_submit:
    directory = f"../RegressionTools/Models/{st.session_state['selected_target']}/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    if name:  # Check if the name is not empty
        regression_streamlit_state = deepcopy(dict(st.session_state))
        genutils.save_object(regression_streamlit_state, f'{name}.pkl', directory)
        st.success(f'Model saved as {name}.pkl')
    else:
        st.error('Please enter a valid model name.')
