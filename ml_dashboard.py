#!/usr/bin/env python
# coding: utf-8

# ## 0. Setting up Environment

# ### 0.1 Libraries

# In[288]:


import pandas as pd
import numpy as np
import re


# ### 0.2 Functions

# In[290]:


# Standardizes strings for columns names
def scrub_colnames(string):
    return re.sub(r'[($)]', '', string.lower().replace(' ', '_')).rstrip("_")


# ## 1. Reading in Data

# ### 1.1 Reading Raw Data

# In[293]:


dat_raw = pd.read_csv('Medicalpremium.csv')


# ### 1.2 Standardize Column Names

# In[295]:


outcols = ['BloodPressureProblems',
           'AnyTransplants',
           'AnyChronicDiseases',
           'KnownAllergies',
           'HistoryOfCancerInFamily',
           'NumberOfMajorSurgeries',
           'PremiumPrice']

incols = ['Blood_Pressure_Problems',
          'Any_Transplants',
          'Any_Chronic_Diseases',
          'Known_Allergies',
          'History_Of_Cancer_In_Family',
          'Number_Of_Major_Surgeries',
          'Premium_Price']

# Replace all columns labels of outcols with incols
dat = dat_raw
for incol, outcol in zip(incols, outcols):
    dat[incol] = dat[outcol]
    dat = dat.drop(outcol, axis=1)
    
dat.columns = dat.columns.map(scrub_colnames)


# ## 3. Predictive Modelling

# ### 3.1 AutoML using Lazy Predict (No Hyper parameter tuning, No Feature Selection, No Cross Validation)

# In[298]:


models = pd.read_csv('data/lpmodels.csv')


# ### 3.3 Top Models as per AutoML

# In[300]:


results_df = pd.read_csv('data/results_df.csv')
# Display the table
print("\nModel Performance Summary:")
print(results_df.to_string(index=False))


# ### 3.5 Best Model

# asets.
# 

# ### SHAP Analysis Commentary – Insurance Premium Prediction - HistGradientBoosting
# 
# This SHAP summary plot explains the impact of each feature on insurance premium predictions:
# 
# #### 1. `age`
# - Higher `age` values (pink dots on far right side of vertical 0 line) significantly increase predicted premiums.
# - Lower `age` values (blue dots on the far left side of vertical 0 line) decrease premiums.
# - Clear positive correlation: **older individuals → higher insurance costs**.
# 
# #### 2. `any_transplants`
# - Having undergone a transplant leads to a strong increase in predicted premiums (pink dots on far right).
# - Those without (blue values - low feature value were not on extremes) have minimal or even negative influence on premiums.
# 
# #### 3. `any_chronic_diseases`
# - Individuals with chronic conditions (pink dots on right side of vertical 0 line) show increase in premiums.
# - Those without (blue values - low feature value were not on extremes) have minimal or even negative influence on premiums.
# 
# #### 4. `number_of_major_surgeries`
# - More major surgeries tend to increase premiums slightly, but the effect is less consistent since low (blue) and high values (pink) overlap around 0
# 
# #### 5. `blood_pressure_problems`
# - Least influential among the top 5 features.
# - High blood pressure (small pink - High Value dots on little right of vertical 0 lines) contributes marginally to higher premiums. The blue and pink dots overlapping around 0 indicate little to no effect on prediction 
# 
# 

# ### 3.8 Scatter Plot of actual Vs predicted premium

# ## 4. Dashboard

# ### 4.1

# In[307]:


import streamlit as st
import altair as alt
import plotly.express as px

#best_model.get_params()
tmodels = models.T
#models


# In[308]:


def make_heatmap(input_df, input_y, input_x, input_color, input_color_theme):
    heatmap = alt.Chart(input_df).mark_rect().encode(
        y=alt.Y(f'{input_y}:O', axis=alt.Axis(title="", titleFontSize=18, titlePadding=15, titleFontWeight=900, labelAngle=0)),
        x=alt.X(f'{input_x}:O', axis=alt.Axis(title="", titleFontSize=18, titlePadding=15, titleFontWeight=900)),
        color=alt.Color(f'max({input_color}):Q',
                        legend=None,
                        scale=alt.Scale(scheme=input_color_theme)),
        stroke=alt.value('black'),
        strokeWidth=alt.value(0.25),
        ).properties(width=900
                     ).configure_axis(
        labelFontSize=12,
        titleFontSize=12
        ) 
    # height=300
    return heatmap

st.set_page_config(
    page_title="US Population Dashboard",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

with st.sidebar:
    st.title('🏂 US Population Dashboard')
    
    model_list = list(models.RMSE.unique())[::-1]
    
    selected_model = st.selectbox('Select a model', model_list, index=len(model_list)-1)
    df_selected_model = models[models.RMSE == selected_model]
    df_selected_model_sorted = df_selected_model.sort_values(by="Time Taken", ascending=False)

    color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
    selected_color_theme = st.selectbox('Select a color theme', color_theme_list)

col = st.columns((5), gap='medium')

with col[0]:
    st.markdown('#### Top States')

    st.dataframe(df_selected_model_sorted,
                 column_order=("RMSE", "Time Taken"),
                 hide_index=True,
                 width=None,
                 column_config={
                    "RMSE": st.column_config.TextColumn(
                        "RMSE",
                    ),
                    "Time Taken": st.column_config.ProgressColumn(
                        "Time Taken",
                        format="%f",
                        min_value=0,
                        max_value=max(df_selected_model_sorted.xs('Time Taken', axis=1)),
                     )}
                 )
    
    with st.expander('About', expanded=True):
        st.write('''
            - Data: [U.S. Census Bureau](<https://www.census.gov/data/datasets/time-series/demo/popest/2010s-state-total.html>).
            - :orange[**Gains/Losses**]: states with high inbound/ outbound migration for selected year
            - :orange[**States Migration**]: percentage of states with annual inbound/ outbound migration > 50,000
            ''')


# In[ ]:




