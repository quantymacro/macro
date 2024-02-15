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

    
if 'selected_target' not in st.session_state:
    st.session_state['selected_target'] = None
if 'models_list' not in st.session_state:
    st.session_state['models_list'] = []
if 'model_files' not in st.session_state:
    st.session_state['model_files'] = []

with st.form(key='SelectTargetVariable'):
    all_targets = os.listdir('../RegressionTools/Models')
    selected_target = st.selectbox('Target Variable', all_targets)
    submit_target = st.form_submit_button('Confirm Target Variable')

if submit_target:
    st.session_state['selected_target'] = selected_target
    models_list = []
    directory = f'../RegressionTools/Models/{st.session_state["selected_target"]}/'
    model_files = os.listdir(directory)
    for file_name in model_files:
        model = genutils.load_object(file_name, directory)
        models_list.append(model)
    st.session_state['models_list'] = models_list
    st.session_state['model_files'] = model_files

# If a target variable has been selected, show the model selection dropdown
if st.session_state['selected_target']:
    model_name = st.selectbox('Model Name', st.session_state['model_files'])
    if model_name:
        idx = st.session_state['model_files'].index(model_name)
        # Assuming stutils.regression_analytics is a function you have to analyze the model
        stutils.regression_analytics(st.session_state['models_list'][idx])