import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import load_model
import xgboost as xgb

# Load Data
@st.cache_data
def load_data():
    data = pd.read_csv(r'C:\Users\prame\OneDrive\Documents\UAP\meat_consumption_worldwide.csv')
    return data

# Load Encoders, Scaler, and Models
location_encoder = joblib.load(r'C:\Users\prame\OneDrive\Documents\UAP\location_encoder.pkl')
measure_encoder = joblib.load(r'C:\Users\prame\OneDrive\Documents\UAP\measure_encoder.pkl')
scaler = joblib.load(r'C:\Users\prame\OneDrive\Documents\UAP\scaler.pkl')
rf_model = joblib.load(r'C:\Users\prame\OneDrive\Documents\UAP\random_forest_model.pkl')

# Load XGBoost Model
xgb_model = xgb.Booster()
xgb_model.load_model(r'C:\Users\prame\OneDrive\Documents\UAP\xgboost_model.json')

# Load Feedforward Neural Network
ff_model = load_model(r'C:\Users\prame\OneDrive\Documents\UAP\feedforward_model.h5')

data = load_data()

st.title("Meat Consumption Analysis and Prediction")

# Exploratory Data Analysis
st.subheader("Dataset Overview")
st.write(data.head())

st.subheader("Data Information")
st.write(data.info())

st.subheader("Data Description")
st.write(data.describe())

# Check for missing values
st.subheader("Missing Values")
st.write(data.isnull().sum())

# Prediction Section
st.subheader("Make Predictions")
scaled_input = None  # Initialize scaled_input as None to avoid reference errors

with st.form("prediction_form"):
    location = st.selectbox("Select Location", options=location_encoder.classes_)
    subject = st.selectbox("Select Subject", options=measure_encoder.classes_)
    time = st.number_input("Enter Year", min_value=int(data['TIME'].min()), max_value=int(data['TIME'].max()), step=1)
    value = st.number_input("Enter Value", min_value=0.0, max_value=float(data['Value'].max()), step=0.1)
    submit = st.form_submit_button("Predict")

    if submit:
        # Encode user input
        location_encoded = location_encoder.transform([location])[0]
        subject_encoded = measure_encoder.transform([subject])[0]

        # Prepare input
        input_data = pd.DataFrame(
            [[location_encoded, subject_encoded, time, value]],
            columns=['LOCATION', 'SUBJECT', 'TIME', 'Value']
        )

        # Debugging: Show raw input data
        st.write("Input data before scaling:")
        st.write(input_data)

        # Scale input data
        scaled_input = scaler.transform(input_data)

        # Debugging: Show scaled input
        st.write("Scaled input:")
        st.write(scaled_input)

        # Predictions
        rf_prediction = rf_model.predict(scaled_input)[0]

        dtest = xgb.DMatrix(scaled_input)
        xgb_raw_prediction = xgb_model.predict(dtest)
        
        # Handle single-class or multi-class output
        if len(xgb_raw_prediction.shape) == 1:  # Single-class output
            xgb_prediction = int(round(xgb_raw_prediction[0]))
        else:  # Multi-class output
            xgb_prediction = np.argmax(xgb_raw_prediction, axis=1)[0]

        ff_raw_prediction = ff_model.predict(scaled_input)
        ff_prediction = np.argmax(ff_raw_prediction, axis=-1)[0]

        # Decode predictions (if applicable)
        rf_prediction_decoded = measure_encoder.inverse_transform([rf_prediction])[0]
        xgb_prediction_decoded = measure_encoder.inverse_transform([xgb_prediction])[0]
        ff_prediction_decoded = measure_encoder.inverse_transform([ff_prediction])[0]

        st.write(f"Random Forest Prediction: {rf_prediction_decoded}")
        st.write(f"XGBoost Prediction: {xgb_prediction_decoded}")
        st.write(f"Feedforward Neural Network Prediction: {ff_prediction_decoded}")

# Display Scaler Details
st.subheader("Scaler Details")

if scaled_input is not None:
    try:
        # Reverse scale values back to original
        reverted_input = scaler.inverse_transform(scaled_input)
        input_data[['TIME', 'Value']] = reverted_input[:, 2:]  # Adjust column indices as needed

        # Debugging: Show reverted input data
        st.write("Reverted input data:")
        st.write(input_data)
    except Exception as e:
        st.error(f"Error saat melakukan inverse scaling: {e}")
else:
    st.write("No scaled input available. Please make a prediction to view scaled input details.")

# Debugging: Display Scaler Attributes
st.subheader("Scaler Debugging")
if hasattr(scaler, "mean_"):
    st.write("Scaler Mean:", scaler.mean_)
if hasattr(scaler, "scale_"):
    st.write("Scaler Scale:", scaler.scale_)
