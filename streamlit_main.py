# streamlit script for running and visualizing the model

# import libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import datetime as dt

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
from PIL import Image

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')
import fx

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
    forecast.drop(columns=["Source_time"], inplace=True)
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

    # slice total_x_test data to only get data up to the forecast datapoint
    total_X_test = total_X_test.loc[:forecast_df.index[0]]

    wind_speed_data = pd.concat([forecast_df["Speed"], total_X_test["Speed"]], axis=0)

    # combine testdata and forecastdf for easy plotting
    final_df = pd.concat([test_data[["predict", "actual"]], forecast_df["Power Generation Forecast"]], axis=0)
    final_df.columns = ["Model", "Actual", "Forecast"]
    return final_df, wind_speed_data

st.title("Wind Power Production Prediction for the Orkney Islands")
st.subheader("By Jannik Elsäßer")
st.caption("This app is a part of my research paper submission for the course 'Large Scale Data Analysis' at the IT University of Copenhagen.")
st.markdown("-------")
st.write("This app uses a combination of artificial neural networks and other machine learning models to predict the power generation of the wind turbines on the Orkney Islands.")
st.write("The weather data used to train the models is from the [MetOffice weather station at Westray Airfield](https://www.metoffice.gov.uk/weather/forecast/gftcsumwq#?date=2023-02-23), and the power generation data used to train the models is from [SSEN.](https://www.ssen.co.uk/our-services/active-network-management/)")

with st.expander("Pipeline diagram:"):
    st.write("For a more detailed explanation of the pipeline, please see the my research paper [submission](https://github.com/jannik-el/OrkneyWindPowerPrediction/blob/main/pdfs/LSDA_Assignment_1FINAL.pdf)")
    diagram = "https://raw.githubusercontent.com/jannik-el/OrkneyWindPowerPrediction/main/figs/ModelDrawing.png"
    st.image(diagram, caption="Pipeline diagram")

st.markdown("-------")

# st.metric the current windspeed and power generation in three columns, set delta to the difference between the second newest data point
# and the newest data point

data = fx.pull_data(days=5)

format="%H:%M"
date = data.index[-1].strftime(format)

st.info(f"The most recent MetOffice weather observation ({date}) from Westray Airfield:", icon="🛰️")
col1, col2, col3 = st.columns(3)
col1.metric("Current Wind Speed [m/s]", str(round(data["Speed"].iloc[-1], 2)), delta=round(data["Speed"].iloc[-1] - data["Speed"].iloc[-2]))
col2.metric("Current Wind Direction [°]", str(direction_map[data["Direction"].iloc[-1]]), delta=round(direction_map[data["Direction"].iloc[-1]] - direction_map[data["Direction"].iloc[-2]]))
col3.metric("Current Power Generation [MW] ", round(data["Total"].iloc[-1], 2), delta=round(data["Total"].iloc[-1] - data["Total"].iloc[-2]))
st.caption("Arrow below is difference to 3H ago")

st.markdown("-------")

with st.container():
    col1, col2 = st.columns(2)
    button = col1.button("Get forecast for the next 5 days from today")
    days = col2.slider("How many days of data should be used to train the model?", min_value=1, max_value=180, value=90, step=1)
    col2.caption("Less days increase streamlit runtime, 90 is best.")
    data = fx.pull_data(days=days)

    if button:
        with st.spinner("Preparing Data..."):
            tscv = TimeSeriesSplit(n_splits=5)

            anm_gridsearch = GridSearchCV(anm_pipeline, anm_params, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
            non_anm_gridsearch = GridSearchCV(non_anm_pipeline, non_anm_params, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)

            ANM_X_train, ANM_y_train, ANM_X_test, ANM_y_test = fx.data_splitting(data, output_val="ANM")
            non_ANM_X_train, non_ANM_y_train, non_ANM_X_test, non_ANM_y_test = fx.data_splitting(data, output_val="Non-ANM")
            total_X_train, total_y_train, total_X_test, total_y_test = fx.data_splitting(data, output_val="Total")

        with st.spinner("Grid searching, this may take a couple of seconds..."):
            non_anm_gridsearch, non_anm_best_params, non_anm_best_score, non_anm_test_score = train_models(non_ANM_X_train, non_ANM_y_train, non_ANM_X_test, non_ANM_y_test, non_anm_gridsearch)
        with st.spinner("Stay with me, we're almost done..."):
            anm_gridsearch, anm_best_params, anm_best_score, anm_test_score = train_models(ANM_X_train, ANM_y_train, ANM_X_test, ANM_y_test, anm_gridsearch)
            pred, total_test_score = predict_and_combine(ANM_X_test, non_ANM_X_test, total_y_test, anm_gridsearch, non_anm_gridsearch)
        
        with st.spinner("Training the best estimator on all data, I swear we're almost there..."):
            anm_model, non_anm_model = load_models_and_train_on_all_data(data, anm_gridsearch, non_anm_gridsearch)
            _, total_test_score = predict_and_combine(ANM_X_test, non_ANM_X_test, total_y_test, anm_model, non_anm_model)

        with st.spinner("Getting the forecast..."):
            forecast = fx.load_forecasts()
            
        with st.spinner("Predicting the future..."):
            total_X_train, total_y_train = fx.final_data_splitting(data, output_val="Total")
            anm_pred = anm_model.predict(forecast)
            non_anm_pred = non_anm_model.predict(forecast)

        with st.spinner("Preparing the results..."):
            forecast_df = create_forecast_df(forecast, anm_pred, non_anm_pred)
            final_df, wind_speed_data = create_final_plotting_df(forecast_df, data)

        st.balloons()
        st.success("All done, thanks for your patience!")

        fig = px.line(final_df, x=final_df.index, y=["Model", "Actual", "Forecast"], title="Power Generation Forecast (Test Data and Predicted Future Power Generation)")
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Power Generation (MW)")
        fig.update_layout(legend_title_text="")
        customdata = wind_speed_data
        fig.update_traces(hovertemplate="<b>Power Generation: %{y:.2f} MW </b><br> Wind Speed: %{customdata:.2f} m/s <br><i> %{x}</i><extra></extra>", customdata=customdata)
        st.plotly_chart(fig, use_container_width=False)

        with st.expander("Model Performance and Parameters"):
            cola, colb, colc = st.columns(3)

            cola.metric("ANM Model MSE Score*", round(anm_test_score, 3))
            colb.metric("Total Model MSE Score*", round(total_test_score, 3))
            colc.metric("Non-ANM Model MSE Score*", round(non_anm_test_score, 3))

            st.caption("*MSE: Mean Squared Error Score, always measured on test data set (split using TimeSeriesSplit(n_splits=5))")

            st.write("ANM Model Best Parameters")
            st.table(anm_best_params)
            st.write("Non-ANM Model Best Parameters")
            st.table(non_anm_best_params)

st.markdown("-------")
itu_logo = "https://en.itu.dk/svg/itu/logo_en.svg"
st.image(itu_logo, use_column_width=True)