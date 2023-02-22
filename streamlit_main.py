# streamlit script for running and visualizing the model

# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# import models
from sklearn.linear_model import LinearRegression, HuberRegressor, RANSACRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit


import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')
import fx

data = fx.pull_data(days=90)

# pipelines
anm_pipeline = Pipeline(steps=[
    ("col_transformer", ColumnTransformer(transformers=[
        ("time", fx.TimestampTransformer(), []),
        ("Speed", None, ["Speed"]),
        ("Direction", None, ["Direction"]),
        ], remainder="drop")),
    ("model", None)
])

anm_params = {
    'col_transformer__Speed': [None, StandardScaler(), PolynomialFeatures(), fx.EmpiricalWaveletTransform(level=5)],
    'col_transformer__Direction': ["drop", fx.WindDirectionMapper(), fx.CompassToCartesianTransformer()],
    'model': [
        LinearRegression(), 
        MLPRegressor(hidden_layer_sizes=(150, 150), activation='tanh', solver='sgd'), 
        SVR(kernel='rbf', gamma='scale', C=1.0, epsilon=0.1),
        HuberRegressor(epsilon=1.35, alpha=0.0001),
        RANSACRegressor(min_samples=0.1, max_trials=100),
        GaussianProcessRegressor(alpha=0.1, kernel=RBF()) 
    ]
}

non_anm_pipeline = Pipeline(steps=[
    ("col_transformer", ColumnTransformer(transformers=[
        ("Speed", None, ["Speed"]),
        ("Direction", None, ["Direction"]),
        ], remainder="drop")),
    ("model", None)
])

non_anm_params = {
    'col_transformer__Speed': [None, StandardScaler(), PolynomialFeatures(), fx.EmpiricalWaveletTransform(level=5)],
    'col_transformer__Direction': ["drop", fx.WindDirectionMapper(), fx.CompassToCartesianTransformer()],
    'model': [
        LinearRegression(), 
        MLPRegressor(hidden_layer_sizes=(150, 150), activation='tanh', solver='sgd'), 
        SVR(kernel='rbf', gamma='scale', C=1.0, epsilon=0.1),
        HuberRegressor(epsilon=1.35, alpha=0.0001),
        RANSACRegressor(min_samples=0.1, max_trials=100),
        GaussianProcessRegressor(alpha=0.1, kernel=RBF()) 
    ]
}

direction_map = {
            "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
            "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
            "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
            "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
        }
###############################

def train_models(X_train, y_train, X_test, y_test, gridsearch):
        gridsearch.fit(X_train, y_train)
        return gridsearch, gridsearch.best_params_, gridsearch.best_score_, gridsearch.score(X_test, y_test)*-1

def predict_and_combine(ANM_X_test, non_ANM_X_test, y_test, anm_gridsearch, non_anm_gridsearch):
        anm_pred = anm_gridsearch.predict(ANM_X_test)
        non_anm_pred = non_anm_gridsearch.predict(non_ANM_X_test)
        pred = anm_pred + non_anm_pred
        return pred, fx.MSE(pred, y_test)


st.title("Wind Power Forecasting on the Orkney Islands")

# st.metric the current windspeed and power generation in three columns, set delta to the difference between the second newest data point
# and the newest data point
col1, col2, col3 = st.columns(3)
col1.metric("Current Wind Speed", str(round(data["Speed"].iloc[-1], 2)) + "[m/s]", delta=str(round(data["Speed"].iloc[-1] - data["Speed"].iloc[-2]))+ "[m/s]")
col2.metric("Current Wind Direction", str(direction_map[data["Direction"].iloc[-1]]) + "°", delta=str(round(direction_map[data["Direction"].iloc[-1]] - direction_map[data["Direction"].iloc[-2]])) + "°")
col3.metric("Current Power Generation", str(round(data["Total"].iloc[-1], 2)) + "[MW]", delta=str(round(data["Total"].iloc[-1] - data["Total"].iloc[-2])) + "[MW]")


# main
with st.expander("Open to see the input data"):
    st.markdown("Input Data table representation")
    st.dataframe(data.head(3))

button = st.button("Run Model")

if button:
    ####### ML Model stuff #######
    tscv = TimeSeriesSplit(n_splits=5)

    anm_gridsearch = GridSearchCV(anm_pipeline, anm_params, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    non_anm_gridsearch = GridSearchCV(non_anm_pipeline, non_anm_params, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)

    ANM_X_train, ANM_y_train, ANM_X_test, ANM_y_test = fx.data_splitting(data, output_val="ANM")
    non_ANM_X_train, non_ANM_y_train, non_ANM_X_test, non_ANM_y_test = fx.data_splitting(data, output_val="Non-ANM")
    total_X_train, total_y_train, total_X_test, total_y_test = fx.data_splitting(data, output_val="Total")

    anm_gridsearch, anm_best_params, anm_best_score, anm_test_score = train_models(ANM_X_train, ANM_y_train, ANM_X_test, ANM_y_test, anm_gridsearch)
    non_anm_gridsearch, non_anm_best_params, non_anm_best_score, non_anm_test_score = train_models(non_ANM_X_train, non_ANM_y_train, non_ANM_X_test, non_ANM_y_test, non_anm_gridsearch)

    
    pred, total_test_score = predict_and_combine(ANM_X_test, non_ANM_X_test, total_y_test, anm_gridsearch, non_anm_gridsearch)

tab1, tab2, tab3 = st.tabs(["Total Model Prediction", "ANM Model", "Non-ANM Model"])