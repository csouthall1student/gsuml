#!/usr/bin/env python
# coding: utf-8

# ## 0. Setting up Environment

# ### 0.1 Libraries

# In[330]:


import pandas as pd
import numpy as np
import re


# ### 0.2 Functions

# In[332]:


# Standardizes strings for columns names
def scrub_colnames(string):
    return re.sub(r'[($)]', '', string.lower().replace(' ', '_')).rstrip("_")


# ## 1. Reading in Data

# ### 1.1 Reading Raw Data

# In[335]:


dat_raw = pd.read_csv('Medicalpremium.csv')


# ### 1.2 Standardize Column Names

# In[337]:


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

# In[340]:


models = pd.read_csv('data/lpmodels.csv')


# ### 3.3 Top Models as per AutoML

# In[342]:


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

# In[349]:


import streamlit as st
import altair as alt
import plotly.express as px


st.set_page_config(
    page_title="Your App Title",
    layout="wide",
    initial_sidebar_state="expanded"
)

alt.themes.enable("quartz")

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


with st.sidebar:
    st.title('🏂 US Population Dashboard')
    
    model_list = list(models.Model.unique())[::-1]
    
    selected_model = st.selectbox('Select a model', model_list, index=len(model_list)-1)
    df_selected_model = models[models.Model == selected_model]
    df_selected_model_sorted = df_selected_model.sort_values(by="RMSE", ascending=False)

    selected_color_theme = 'blues'

col = st.columns((15, 15), gap='medium')

with col[0]:
    st.markdown('#### LazyPredict Results')

    st.dataframe(df_selected_model_sorted,
                 column_order=("Model",
                               "RMSE",
                               "R-Squared",
                               "Adjusted R-Squared",
                               "Time Taken"),
                 hide_index=True,
                 width=None,
                 column_config={
                    "Model": st.column_config.TextColumn(
                        "Model",
                    ),
                    "RMSE": st.column_config.ProgressColumn(
                        "RMSE",
                        format="%f",
                        min_value=0,
                        max_value=max(df_selected_model_sorted.xs('RMSE', axis=1))
                     ),
                     "R-Squared": st.column_config.ProgressColumn(
                        "R-Squared",
                        format="%f",
                        min_value=0,
                        max_value=max(df_selected_model_sorted.xs('RMSE', axis=1))
                     ),
                     "Adjusted R-Squared": st.column_config.ProgressColumn(
                        "Adjusted R-Squared",
                        format="%f",
                        min_value=0,
                        max_value=max(df_selected_model_sorted.xs('RMSE', axis=1))
                     ),
                     "Time Taken": st.column_config.ProgressColumn(
                        "Time Taken",
                        format="%f",
                        min_value=0,
                        max_value=max(df_selected_model_sorted.xs('RMSE', axis=1))
                     )}
                 )
    
with st.expander('About', expanded=True):
    st.write('''
        - Data: [Kaggle Medical Insurance Premium](<https://www.kaggle.com/datasets/tejashvi14/medical-insurance-premium-prediction/data>).
        - :orange[**Age**]: Age of customer.
        - :orange[**Height**]: Height of customer.
        - :orange[**Weight**]: Weight of customer.
        - :orange[**Diabetes**]: Whether the person has abnormal blood sugar levels.
        - :orange[**Blood Pressure Problems**]: Whether the person Has abnormal blood pressure levels.
        - :orange[**Any Transplants**]: Any major organ transplants.
        - :orange[**Any Chronice Disease**]: Whether customer suffers from chronic ailments like asthama, etc.
        - :orange[**Known Allergies**]: Whether the customer has any known allergies.
        - :orange[**History of Cancer**]: Whether any blood relative of the customer has had any form of cancer.
        - :orange[**Number of Major Surgeries**]: The number of major surgeries that the person has had.
        - :green[**Premium Price**]: Target variable for prediction to create a model that predicts the yearly medical cover cost
        ''')


# In[ ]:




