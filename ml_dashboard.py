#!/usr/bin/env python
# coding: utf-8

# ## 0. Setting up Environment

# ### 0.1 Libraries

# In[330]:


import pandas as pd
import numpy as np
import re

import streamlit as st
import altair as alt
import plotly.express as px

import shap
from streamlit_shap import st_shap
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import HistGradientBoostingRegressor

# Reading in data used in shap plot generation
X_train = pd.read_csv('data/X_train_shap.csv')
X_test = pd.read_csv('data/X_test_shap.csv')
y_train = pd.read_csv('data/y_train_shap.csv')

# Shap plot generation
# Step 1: Feature Selection with names preserved
selector = SelectKBest(score_func=f_regression, k=5)
selector.fit(X_train, y_train)

# Get selected feature names
selected_features = X_train.columns[selector.get_support()]

# Transform train/test sets using selected columns only
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Step 2: Re-train model with selected features
best_model = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, random_state=42)
best_model.fit(X_train_selected, y_train)

# Step 3: SHAP Explanation
explainer = shap.Explainer(best_model, X_train_selected)
shap_values = explainer(X_test_selected)

# Reading in our processed data
models = pd.read_csv('data/lpmodels.csv')
results_df = pd.read_csv('data/results_df.csv')

#Setting up streamlit page
st.set_page_config(
    page_title="Your App Title",
    layout="wide",
    initial_sidebar_state="expanded"
)

#Setting up streamlit sidebar
with st.sidebar:
    st.title('Fine Tuned Model Metrics')
    measure_list = ['RMSE', 'R²', 'Adjusted R²', 'CV RMSE', 'CV R²', 'CV Adjusted R²']
    select_measure = st.selectbox('Select a measure', measure_list, index=len(measure_list)-1)

#Setting up some columns for making smaller graphic windows
col = st.columns((15, 15), gap='small')

#Setting up our bar chart with error metrics across models
with col[0]:
    st.markdown('#### Fine-Tuned Model Error')

    st.altair_chart(alt.Chart(results_df)
                    .mark_bar()
                    .encode(
                        x='Model:O',
                        y=f'{select_measure}:Q',
                        color=alt.Color('Model:N', legend=None)
                        )
                    .properties(
                        width=500,
                        height=500
                        )
                    )

#setting up our lazypredictor table display
with col[1]:
    st.markdown('#### LazyPredict Results')

    st.dataframe(models,
                 column_order=("Model",
                               "RMSE",
                               "R-Squared",
                               "Adjusted R-Squared",
                               "Time Taken"),
                 hide_index=True,
                 width=None,
                 column_config={
                    "Model": st.column_config.TextColumn(
                        "Model"
                    ),
                    "RMSE": st.column_config.NumberColumn(
                        "RMSE",
                        format="%.2f"
                     ),
                     "R-Squared": st.column_config.NumberColumn(
                        "R-Squared",
                        format="%.3f"
                     ),
                     "Adjusted R-Squared": st.column_config.NumberColumn(
                        "Adjusted R-Squared",
                        format="%.3f"
                     ),
                     "Time Taken": st.column_config.NumberColumn(
                        "Time Taken",
                        format="%.3f"
                     )}
                 )

# SHAP explanation section
with st.expander('SHAP Analysis Commentary – Insurance Premium Prediction - HistGradientBoosting', expanded=True):
    st.markdown("""
 
 #### 1. `age`
 - Higher `age` values (pink dots on far right side of vertical 0 line) significantly increase predicted premiums.
 - Lower `age` values (blue dots on the far left side of vertical 0 line) decrease premiums.
 - Clear positive correlation: **older individuals → higher insurance costs**.
 
 #### 2. `any_transplants`
 - Having undergone a transplant leads to a strong increase in predicted premiums (pink dots on far right).
 - Those without (blue values - low feature value were not on extremes) have minimal or even negative influence on premiums.
 
 #### 3. `any_chronic_diseases`
 - Individuals with chronic conditions (pink dots on right side of vertical 0 line) show increase in premiums.
 - Those without (blue values - low feature value were not on extremes) have minimal or even negative influence on premiums.
 
 #### 4. `number_of_major_surgeries`
 - More major surgeries tend to increase premiums slightly, but the effect is less consistent since low (blue) and high values (pink) overlap around 0
 
 #### 5. `blood_pressure_problems`
 - Least influential among the top 5 features.
 - High blood pressure (small pink - High Value dots on little right of vertical 0 lines) contributes marginally to higher premiums. The blue and pink dots overlapping around 0 indicate little to no effect on prediction""")

# SHAP plot
st_shap(shap.plots.beeswarm(shap_values), height=500)

# Data link and field explanation
with st.expander('About the Data', expanded=False):
    st.write('''
        - :red[**Data**]: [Kaggle Medical Insurance Premium](<https://www.kaggle.com/datasets/tejashvi14/medical-insurance-premium-prediction/data>).
        - :orange[**Age**]: Age of customer.
        - :orange[**Height**]: Height of customer.
        - :orange[**Weight**]: Weight of customer.
        - :orange[**Diabetes**]: Whether the person has abnormal blood sugar levels.
        - :orange[**Blood Pressure Problems**]: Whether the person has abnormal blood pressure levels.
        - :orange[**Any Transplants**]: Any major organ transplants.
        - :orange[**Any Chronice Disease**]: Whether customer suffers from chronic ailments like asthama, etc.
        - :orange[**Known Allergies**]: Whether the customer has any known allergies.
        - :orange[**History of Cancer**]: Whether any blood relative of the customer has had any form of cancer.
        - :orange[**Number of Major Surgeries**]: The number of major surgeries that the person has had.
        - :green[**Premium Price**]: Target variable for prediction to create a model that predicts the yearly medical cover cost
        ''')



# See you later, Space Cowboy...
