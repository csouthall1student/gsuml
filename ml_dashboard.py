#!/usr/bin/env python
# coding: utf-8

# ## 0. Setting up Environment

# ### 0.1 Libraries

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from tabulate import tabulate


# ### 0.2 Functions

# In[5]:


# Prints small readable dataframes
def pprint_df(dframe):
    print(tabulate(dframe, headers='keys', tablefmt='psql', showindex=False))

# Standardizes strings for columns names
def scrub_colnames(string):
    return re.sub(r'[($)]', '', string.lower().replace(' ', '_')).rstrip("_")


# ## 1. Reading in Data

# ### 1.1 Reading Raw Data

# In[8]:


dat_raw = pd.read_csv('Medicalpremium.csv')


# ### 1.2 Standardize Column Names

# In[10]:


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


# ## 1.3 Generate Descriptive Statistics for Features

# In[12]:


desc = dat.describe()
desc


# ## 2. Data Exploration

# ### 2.1 Continuous/Discrete Numerical Data

# #### 2.1.1 Premium vs. Age

# In[16]:


col = 'age'

bins = [0, 25, 35, 45, 55, 100]
labels = ['<25', '25-35', '35-45', '45-55', '55+']
dat[f'binned_{col}'] = pd.cut(dat[col], bins=bins, labels=labels, right=False)

bar = dat.groupby(f'binned_{col}', observed=True).mean().reset_index()
bar['premium_stddev'] = dat.groupby(f'binned_{col}', observed=True).std().reset_index()['premium_price']
pprint_df(bar[[f'binned_{col}', 'premium_price', 'premium_stddev']])

plt.bar(bar[f'binned_{col}'], bar['premium_price'], color=['#90EE90', '#77DD77', '#32CD32', '#208B22', '#008000'], yerr=bar['premium_stddev'], capsize=10)
plt.title(f'Average Premium by {col}')
plt.xlabel(col)
plt.ylabel('Premium')

dat = dat.drop(f'binned_{col}', axis=1)


# #### 2.1.2 Premium vs. Height

# In[18]:


col = 'height'

bins = [0, desc[col]['25%'], desc[col]['50%'], desc[col]['75%'], desc[col]['max']]
labels = ['0-25%', '25-50%', '50-75%', '75-100%']
dat[f'binned_{col}'] = pd.cut(dat[col], bins=bins, labels=labels, right=False)


bar = dat.groupby(f'binned_{col}', observed=True).mean().reset_index()
bar['premium_stddev'] = dat.groupby(f'binned_{col}', observed=True).std().reset_index()['premium_price']
pprint_df(bar[[f'binned_{col}', 'premium_price', 'premium_stddev']])

plt.bar(bar[f'binned_{col}'], bar['premium_price'], color=['#90EE90', '#32CD32', '#208B22', '#008000'], yerr=bar['premium_stddev'], capsize=10)
plt.title(f'Average Premium by {col}')
plt.xlabel(col)
plt.ylabel('Premium')

dat = dat.drop(f'binned_{col}', axis=1)


# #### 2.1.3 Premium vs. Weight

# In[20]:


col = 'weight'

bins = [0, desc[col]['25%'], desc[col]['50%'], desc[col]['75%'], desc[col]['max']]
labels = ['0-25%', '25-50%', '50-75%', '75-100%']
dat[f'binned_{col}'] = pd.cut(dat[col], bins=bins, labels=labels, right=False)


bar = dat.groupby(f'binned_{col}', observed=True).mean().reset_index()
bar['premium_stddev'] = dat.groupby(f'binned_{col}', observed=True).std().reset_index()['premium_price']
pprint_df(bar[[f'binned_{col}', 'premium_price', 'premium_stddev']])

plt.bar(bar[f'binned_{col}'], bar['premium_price'], color=['#90EE90', '#32CD32', '#208B22', '#008000'], yerr=bar['premium_stddev'], capsize=10)
plt.title(f'Average Premium by {col}')
plt.xlabel(col)
plt.ylabel('Premium')

dat = dat.drop(f'binned_{col}', axis=1)


# #### 2.1.4 Premium vs. Number of Surgeries

# In[22]:


col = 'number_of_major_surgeries'

bar = dat.groupby('number_of_major_surgeries', observed=True).mean().reset_index()

plt.bar(bar['number_of_major_surgeries'], bar['premium_price'], color=['#90EE90', '#32CD32', '#208B22', '#008000'])
plt.title(f'Average Premium by {col}')
plt.xlabel(col)
plt.ylabel('Premium')


# ### 2.2 Binary Variables
# #### Diabetes:
# ##### We observe that subpopulation paying above the 50th percentile premium has greater instances of diabetes.
# #### Blood Pressure Problems:
# ##### We generally observe higher instances of blood pressure problems in quartiles paying greater premiums.
# #### Transplants:
# ##### Of the subpopulation with transplants, over half are paying within the top quartile of premium rates.

# In[24]:


# Binning Premium Data Based on Quartiles
bins = [0, 21000, 23000, 28000, 100000]
labels = ['0-25%', '25-50%', '50-75%', '75-100%']
dat['premium_quartile'] = pd.cut(dat['premium_price'], bins=bins, labels=labels, right=False)

# Fitting Binary Variables into Premium Bins
binary_cols = ['premium_quartile', 'diabetes', 'blood_pressure_problems', 'any_transplants', 'any_chronic_diseases', 'known_allergies', 'history_of_cancer_in_family']
bar2 = dat[binary_cols].groupby('premium_quartile', observed=True).mean().reset_index()

fig, ax = plt.subplots(2,3)
fig.set_figwidth(15); fig.set_figheight(10)

# Supblot for 'diabetes'
ax[0,0].bar_label(ax[0,0].bar(bar2['premium_quartile'], bar2['diabetes'], color='red'),
                 fmt='%.2f')
ax[0,0].set_title('diabetes')
ax[0,0].set_ylabel('Mean')
ax[0,0].set_ylim(0,1)
ax[0,0].set_xlabel('Percentile of Premium Price')

# Supblot for 'blood_pressure_problems'
ax[0,1].bar_label(ax[0,1].bar(bar2['premium_quartile'], bar2['blood_pressure_problems'], color='green'),
                 fmt='%.2f')
ax[0,1].set_title('blood_pressure_problems')
ax[0,1].set_ylabel('Mean')
ax[0,1].set_ylim(0,1)
ax[0,1].set_xlabel('Percentile of Premium Price')

# Supblot for 'any_transplants'
ax[0,2].bar_label(ax[0,2].bar(bar2['premium_quartile'], bar2['any_transplants'], color='blue'),
                 fmt='%.2f')
ax[0,2].set_title('any_transplants')
ax[0,2].set_ylabel('Mean')
ax[0,2].set_ylim(0,1)
ax[0,2].set_xlabel('Percentile of Premium Price')

# Supblot for 'any_chronic_diseases'
ax[1,0].bar_label(ax[1,0].bar(bar2['premium_quartile'], bar2['any_chronic_diseases'], color='orange'),
                 fmt='%.2f')
ax[1,0].set_title('any_chronic_diseases')
ax[1,0].set_ylabel('Mean')
ax[1,0].set_ylim(0,1)
ax[1,0].set_xlabel('Percentile of Premium Price')

# Supblot for 'known_allergies'
ax[1,1].bar_label(ax[1,1].bar(bar2['premium_quartile'], bar2['known_allergies'], color='purple'),
                 fmt='%.2f')
ax[1,1].set_title('known_allergies')
ax[1,1].set_ylabel('Mean')
ax[1,1].set_ylim(0,1)
ax[1,1].set_xlabel('Percentile of Premium Price')

# Supblot for 'history_of_cancer_in_family'
ax[1,2].bar_label(ax[1,2].bar(bar2['premium_quartile'], bar2['history_of_cancer_in_family'], color='gold'),
                 fmt='%.2f')
ax[1,2].set_title('history_of_cancer_in_family')
ax[1,2].set_ylabel('Mean')
ax[1,2].set_ylim(0,1)
ax[1,2].set_xlabel('Percentile of Premium Price')

plt.show()


# ## 3. Predictive Modelling

# In[26]:


dat = dat.drop(columns=['premium_quartile'])  # predictors
# minor data cleaning for predictive modelling


# ### 3.1 AutoML using Lazy Predict (No Hyper parameter tuning, No Feature Selection, No Cross Validation)

# In[28]:


from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split

# Separate features and target
X = dat.drop(columns=['premium_price'])  # predictors
y = dat['premium_price']  # regression target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and run LazyRegressor
reg = LazyRegressor(verbose=0, ignore_warnings=True)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

# Show the results
print(models)


# ### 3.3 Top Models as per AutoML

# ### Top 3 Models from LazyPredict
# 1. HistGradientBoostingRegressor  
# 2. LGBMRegressor  
# 3. RandomForestRegressor
# 
# ![image.png](attachment:43e6db61-8770-48aa-97c4-e5263b12a2e0.png)
# 
# ### Top 3 Models from PyCaret
# 1. Gradient Boosting Regressor  
# 2. LGBMRegressor  
# 3. CatBoost Regressor
# 
# ![image.png](attachment:a2d3cea5-ad96-498a-a1d9-e6d0c3d075f8.png)
# 
# ### Model Comparison Approach followed
# 
# To ensure a fair comparison, all selected models above were re-trained and re-evaluated.
# 
# **Evaluation Metrics:**
# - Cross Validation RMSE (Root Mean Squared Error) – lower is better    
# - Cross Validation Adjusted R² – accounts for model complexity - higher is better
# 
# **Note:**  
# R² values from PyCaret were observed to be lower than those from LazyPredict.  
# This is expected, as PyCaret performs feature selection, hyperparameter tuning and cross validation, which improve generalization but may reduce apparent performance compared to LazyPredict, which uses default model settings without tuning.
# 

# In[35]:


from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline

# Define adjusted R² function
def adjusted_r2_score(r2, n, k):
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

# Define RMSE scorer for CV
rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False, squared=False)

results = {}

# Model grid with basic parameter search space
model_grid = {
    'GradientBoosting': (
        GradientBoostingRegressor(random_state=42),
        {'model__n_estimators': [100, 200], 'model__learning_rate': [0.1, 0.05]}
    ),
    'LGBM': (
        LGBMRegressor(random_state=42),
        {'model__n_estimators': [100, 200], 'model__learning_rate': [0.1, 0.05]}
    ),
    'CatBoost': (
        CatBoostRegressor(verbose=0, random_state=42),
        {'model__depth': [4, 6], 'model__learning_rate': [0.1, 0.05]}
    ),
    'HistGradientBoosting': (
        HistGradientBoostingRegressor(random_state=42),
        {'model__max_iter': [100, 200], 'model__learning_rate': [0.1, 0.05]}
    ),
    'RandomForest': (
        RandomForestRegressor(random_state=42),
        {'model__n_estimators': [100, 200], 'model__max_depth': [None, 10]}
    )
}

# Loop through all models
for name, (model, param_grid) in model_grid.items():
    print(f"Training {name}...")

    pipeline = Pipeline([
        ('feature_selection', SelectKBest(score_func=f_regression, k=5)),
        ('model', model)
    ])

    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    n = X_test.shape[0]
    k = X_test.shape[1]

    # Hold-out test metrics
    mse = mean_squared_error(y_test, y_pred)#, squared=False)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    adj_r2 = adjusted_r2_score(r2, n, k)

    # Cross-validation scores
    cv_r2_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
    cv_r2 = np.mean(cv_r2_scores)
    cv_adj_r2 = adjusted_r2_score(cv_r2, X_train.shape[0], k)

    cv_rmse_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring=rmse_scorer)
    cv_rmse = -np.mean(cv_rmse_scores)

    results[name] = {
        'Best Params': grid.best_params_,
        'RMSE': rmse,
        'R²': r2,
        'Adjusted R²': adj_r2,
        'CV RMSE': cv_rmse,
        'CV R²': cv_r2,
        'CV Adjusted R²': cv_adj_r2
    }

# Display results in tabular format
results_df = pd.DataFrame(results).T
results_df = results_df[['RMSE', 'R²', 'Adjusted R²', 'CV RMSE', 'CV R²', 'CV Adjusted R²', 'Best Params']]
print("\nModel Performance Comparison with Feature Selection & Tuning:")
print(results_df.round(3))


# In[36]:


# Convert results dictionary to DataFrame
results_df = pd.DataFrame(results).T  # Transpose for model names as rows
results_df = results_df.reset_index().rename(columns={'index': 'Model'})

# Round numeric columns for display
numeric_cols = ['RMSE', 'R²', 'Adjusted R²', 'CV R²', 'CV Adjusted R²']
results_df[numeric_cols] = results_df[numeric_cols].round(3)

# Display the table
print("\nModel Performance Summary:")
print(results_df.to_string(index=False))


# ### 3.4 Compare Top Machine Learning Models

# In[39]:


# Convert results to DataFrame and sort by R²
results_df = pd.DataFrame(results).T.sort_values(by='R²', ascending=False)

# Set up figure with 6 subplots
fig, ax = plt.subplots(2, 3, figsize=(21, 10))

# RMSE plot
results_df['RMSE'].plot(kind='barh', ax=ax[0, 0], color='coral')
ax[0, 0].set_title('RMSE (Lower is Better)')
ax[0, 0].set_xlabel('RMSE')
ax[0, 0].invert_yaxis()
ax[0, 0].grid(True, linestyle='--', alpha=0.6)

# R² plot
results_df['R²'].plot(kind='barh', ax=ax[0, 1], color='seagreen')
ax[0, 1].set_title('R² Score (Higher is Better)')
ax[0, 1].set_xlabel('R²')
ax[0, 1].invert_yaxis()
ax[0, 1].grid(True, linestyle='--', alpha=0.6)

# Adjusted R² plot
results_df['Adjusted R²'].plot(kind='barh', ax=ax[0, 2], color='slateblue')
ax[0, 2].set_title('Adjusted R² (Higher is Better)')
ax[0, 2].set_xlabel('Adjusted R²')
ax[0, 2].invert_yaxis()
ax[0, 2].grid(True, linestyle='--', alpha=0.6)

# CV RMSE plot
results_df['CV RMSE'].plot(kind='barh', ax=ax[1, 0], color='darkorange')
ax[1, 0].set_title('CV RMSE (Lower is Better)')
ax[1, 0].set_xlabel('CV RMSE')
ax[1, 0].invert_yaxis()
ax[1, 0].grid(True, linestyle='--', alpha=0.6)

# CV R² plot
results_df['CV R²'].plot(kind='barh', ax=ax[1, 1], color='darkgreen')
ax[1, 1].set_title('CV R² (Higher is Better)')
ax[1, 1].set_xlabel('CV R²')
ax[1, 1].invert_yaxis()
ax[1, 1].grid(True, linestyle='--', alpha=0.6)

# CV Adjusted R² plot
results_df['CV Adjusted R²'].plot(kind='barh', ax=ax[1, 2], color='mediumslateblue')
ax[1, 2].set_title('CV Adjusted R² (Higher is Better)')
ax[1, 2].set_xlabel('CV Adjusted R²')
ax[1, 2].invert_yaxis()
ax[1, 2].grid(True, linestyle='--', alpha=0.6)

plt.suptitle('Model Performance Comparison (Hold-Out vs CV)', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# ### 3.5 Best Model

# asets.
# 

# ### Why We Selected `HistGradientBoostingRegressor` as the Champion Model
# 
# - It achieved the **third-lowest Cross-Validated RMSE**, indicating strong predictive performance, only slightly behind the top two models.
# - It also secured the **third-highest Cross-Validated Adjusted R²**, demonstrating a good balance between accuracy and model complexity.
# - We preferred it over `GradientBoostingRegressor` because:
#   - `HistGradientBoostingRegressor` is **faster and more efficient**, especially on large datasets, which is important for this use case.
# - We selected it over `LGBMRegressor` because:
#   - `LGBMRegressor` can be **more prone to overfitting**
# 

# ### 3.6 Tradeoffs Made while selecting the Best Model

# ### Summary Tradeoff Table:
# 
# ![image.png](attachment:1b9afa8c-f82f-4e80-a62a-a496cd83054c.png)
# 

# ### 3.7 SHAP Plot for Best Model

# In[41]:


import shap
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import HistGradientBoostingRegressor

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

# Step 4: SHAP Beeswarm Plot with actual feature names
shap.plots.beeswarm(shap_values, max_display=10)


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

# In[43]:


import shap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import HistGradientBoostingRegressor

# Step 1: Feature Selection
selector = SelectKBest(score_func=f_regression, k=5)
selector.fit(X_train, y_train)

# Get selected feature names
selected_features = X_train.columns[selector.get_support()]

# Transform train/test sets using selected columns
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Step 2: Fit HistGradientBoostingRegressor
best_model = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, random_state=42)
best_model.fit(X_train_selected, y_train)

# Step 3: Predict on test set
y_pred = best_model.predict(X_test_selected)

# Step 4: Plot Actual vs Predicted
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, color='blue', edgecolor='black', alpha=0.6)

# Ideal line (perfect prediction)
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

plt.title('Scatter Plot: Actual vs Predicted (HistGradientBoostingRegressor)', fontsize=14)
plt.xlabel('Actual Insurance Premium (Y_test)', fontsize=12)
plt.ylabel('Predicted Insurance Premium (Y_pred)', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()


# #### The Actual and Predicted value scatter points fall along the 45 degree line which reflects good predictability

# ## 4. Dashboard

# ### 4.1

# In[69]:


import streamlit as st
import altair as alt
import plotly.express as px

#best_model.get_params()
tmodels = models.T
models


# In[85]:


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




