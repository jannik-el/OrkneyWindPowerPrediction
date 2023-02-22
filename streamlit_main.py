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

def load_models_and_train_on_all_data(data, anm_gridsearch, non_anm_gridsearch):
    # naming schemes is not my strong suit
    anm_X_train, anm_y_train = fx.final_data_splitting(data, output_val="ANM")
    non_anm_X_train, non_anm_y_train = fx.final_data_splitting(data, output_val="Non-ANM")
    anm_gridsearch.fit(anm_X_train, anm_y_train)
    non_anm_gridsearch.fit(non_anm_X_train, non_anm_y_train)
    return anm_gridsearch, non_anm_gridsearch

def create_forecast_df(forecast, anm_pred, non_anm_pred):
    future = anm_pred + non_anm_pred
    forecast["Power Generation Forecast"] = future
    forecast = forecast.resample("3H").mean()
    forecast.drop(columns=["Speed", "Source_time"], inplace=True)
    return forecast

def create_final_plotting_df(forecast_df, data):
    # this code is just for plotting the final graph
    ANM_X_train, ANM_y_train, ANM_X_test, ANM_y_test = fx.data_splitting(data, output_val="ANM")
    non_ANM_X_train, non_ANM_y_train, non_ANM_X_test, non_ANM_y_test = fx.data_splitting(data, output_val="Non-ANM")
    total_X_train, total_y_train, total_X_test, total_y_test = fx.data_splitting(data, output_val="Total")

    test_anm_pred = anm_model.predict(ANM_X_test)
    test_non_anm_pred = non_anm_model.predict(non_ANM_X_test)

    test_prediction = test_anm_pred + test_non_anm_pred
    test_data = fx.create_timestamps(test_prediction, total_X_test, total_y_test)

    # combine testdata and forecastdf for easy plotting
    final_df = pd.concat([test_data, forecast_df], axis=0)

    final_df.columns = ["Model", "Actual", "Forecast"]
    return final_df

st.title("Wind Power Forecasting on the Orkney Islands")

# st.metric the current windspeed and power generation in three columns, set delta to the difference between the second newest data point
# and the newest data point

st.write("The current weather in the orkneys:")
col1, col2, col3 = st.columns(3)
col1.metric("Current Wind Speed [m/s]", str(round(data["Speed"].iloc[-1], 2)), delta=round(data["Speed"].iloc[-1] - data["Speed"].iloc[-2]))
col2.metric("Current Wind Direction [°]", str(direction_map[data["Direction"].iloc[-1]]), delta=round(direction_map[data["Direction"].iloc[-1]] - direction_map[data["Direction"].iloc[-2]]))
col3.metric("Current Power Generation [MW] ", round(data["Total"].iloc[-1], 2), delta=round(data["Total"].iloc[-1] - data["Total"].iloc[-2]))
st.caption("Arrow below is difference to 3H ago")


# main
with st.expander("Open to see the input data"):
    st.markdown("Input Data table representation")
    st.dataframe(data.head(3))

button = st.button("Run Model")

if button:
    with st.spinner("Preparing Data..."):
        tscv = TimeSeriesSplit(n_splits=5)

        anm_gridsearch = GridSearchCV(anm_pipeline, anm_params, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
        non_anm_gridsearch = GridSearchCV(non_anm_pipeline, non_anm_params, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)

        ANM_X_train, ANM_y_train, ANM_X_test, ANM_y_test = fx.data_splitting(data, output_val="ANM")
        non_ANM_X_train, non_ANM_y_train, non_ANM_X_test, non_ANM_y_test = fx.data_splitting(data, output_val="Non-ANM")
        total_X_train, total_y_train, total_X_test, total_y_test = fx.data_splitting(data, output_val="Total")

    with st.spinner("Grid Searching, this may take a couple of seconds..."):
        anm_gridsearch, anm_best_params, anm_best_score, anm_test_score = train_models(ANM_X_train, ANM_y_train, ANM_X_test, ANM_y_test, anm_gridsearch)
        non_anm_gridsearch, non_anm_best_params, non_anm_best_score, non_anm_test_score = train_models(non_ANM_X_train, non_ANM_y_train, non_ANM_X_test, non_ANM_y_test, non_anm_gridsearch)

        pred, total_test_score = predict_and_combine(ANM_X_test, non_ANM_X_test, total_y_test, anm_gridsearch, non_anm_gridsearch)
    
    with st.spinner("Training on all data, this will probably also take some seconds..."):
        anm_model, non_anm_model = load_models_and_train_on_all_data(data, anm_gridsearch, non_anm_gridsearch)

    with st.spinner("Getting the forecast..."):
        forecast = fx.load_forecasts()
        
    with st.spinner("Predicting the future..."):
        total_X_train, total_y_train = fx.final_data_splitting(data, output_val="Total")
        anm_pred = anm_model.predict(forecast)
    with st.spinner("Stay with me, we're almost done..."):
        non_anm_pred = non_anm_model.predict(forecast)

    with st.spinner("Preparing the results, I swear this is the second last step..."):
        forecast_df = create_forecast_df(forecast, anm_pred, non_anm_pred)
        final_df = create_final_plotting_df(forecast_df, data)

    st.success("All done, thanks for your patience!")

    tab1, tab2, tab3 = st.tabs(["Total Model Prediction", "ANM Model", "Non-ANM Model"])
    # plot final_df using plotly
    fig = px.line(final_df, x=final_df.index, y=["Model", "Actual", "Forecast"], title="Power Generation Forecast (Test Data and Predicted Future Power Generation)")
    # add x and y axis labels
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Power Generation (MW)")
    # change legend heading
    fig.update_layout(legend_title_text="")
    tab1.plotly_chart(fig)
