
import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("maize_yield_model (1).pkl")
scaler = joblib.load("scaler (1).pkl")

st.title("🌽 Maize Yield Prediction App")
st.markdown("Enter farm conditions below to predict maize yield (tons/ha):")

# Input fields
rainfall = st.number_input("Rainfall (mm)", min_value=0.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0)
soil_pH = st.number_input("Soil pH", min_value=0.0)
clay = st.number_input("Soil Clay (%)", min_value=0.0)
silt = st.number_input("Soil Silt (%)", min_value=0.0)
sand = st.number_input("Soil Sand (%)", min_value=0.0)
fertilizer = st.number_input("Fertilizer Use (kg/ha)", min_value=0.0)
planting_density = st.number_input("Planting Density (plants/m²)", min_value=0.0)

# Predict
if st.button("Predict Yield"):
    input_data = np.array([[rainfall, temperature, soil_pH, clay, silt, sand, fertilizer, planting_density]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    st.success(f"Estimated Maize Yield: {prediction:.2f} tons/ha")
