
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# --- Load the trained model and artifacts ---
@st.cache_resource
def load_model():
    model = joblib.load('hybrid_model.pkl')
    scaler = joblib.load('scaler.pkl')
    with open('feature_cols.txt', 'r') as f:
        feature_cols = f.read().splitlines()
    return model, scaler, feature_cols

model, scaler, feature_cols = load_model()

# --- The "API" Prediction Function ---
def predict_shelf_life(temp, humidity, ethylene, day):
    input_df = pd.DataFrame([{'temperature_C': temp, 'humidity_percent': humidity, 'ethylene_ppm': ethylene, 'day': day}])
    
    # This is the same feature engineering from your training script
    input_df['day_squared'] = input_df['day'] ** 2
    input_df['day_cubed'] = input_df['day'] ** 3
    input_df['temp_humidity_interaction'] = input_df['temperature_C'] * input_df['humidity_percent']
    input_df['temp_ethylene_interaction'] = input_df['temperature_C'] * input_df['ethylene_ppm']
    input_df['temp_normalized'] = (input_df['temperature_C'] - 20) / 10
    input_df['humidity_normalized'] = (input_df['humidity_percent'] - 75) / 25
    input_df['arrhenius_factor'] = np.exp(0.08 * (input_df['temperature_C'] - 20))
    input_df['humidity_stress'] = np.abs(input_df['humidity_percent'] - 87.5) / 50.0
    print("FEATURES THE APP IS LOOKING FOR:", feature_cols)
    print("COLUMNS THE APP ACTUALLY HAS:", list(input_df.columns))
    input_processed = input_df[feature_cols]
    input_scaled = scaler.transform(input_processed)
    prediction = model.predict(input_scaled)[0]
    return np.clip(prediction, 0, 14)

# --- The Aesthetic Frontend ---
st.set_page_config(page_title=" Twinsie 🍌", page_icon="🍌", layout="centered")

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #e67e22;
        text-align: center;
        font-weight: 700;
    }
    .sub-header {
        text-align: center;
        color: #7f8c8d;
    }
    .metric-card {
        background-color: #fff3b2;
        padding: 1.5rem;
        border-radius: 1rem;
        border: 2px solid #f39c12;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .stMetric > div > div > div > div {
        background-color: #fff3b2;
        border: 2px solid #f39c12;
        padding-bottom: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Twinsie</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">your personal ai-powered banana ripeness predictor</p>', unsafe_allow_html=True)

st.sidebar.header("🔧 Sensor Controls")
temp = st.sidebar.slider("🌡️ Temperature (°C)", 15.0, 30.0, 22.0)
humidity = st.sidebar.slider("💧 Humidity (%)", 60.0, 95.0, 85.0)
ethylene = st.sidebar.slider("🍈 Ethylene (ppm)", 5.0, 50.0, 20.0)
day = st.sidebar.slider("📅 Day of Lifecycle", 0.5, 12.0, 3.0)

if st.sidebar.button("🔮 Predict Shelf Life"):
    with st.spinner('Thinking like a banana...'):
        time.sleep(1) # For dramatic effect
        shelf_life = predict_shelf_life(temp, humidity, ethylene, day)
        
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("🌡️ Temp", f"{temp:.1f}°C")
        col2.metric("💧 Humidity", f"{humidity:.0f}%")
        col3.metric("🍈 Ethylene", f"{ethylene:.1f} ppm")
        col4.metric("⏳ Shelf Life", f"{shelf_life:.1f} days")
        
        st.markdown("---")
        
        # The Big Banana Visual
        if shelf_life > 5:
            st.markdown("<h2 style='text-align: center; color: #27ae60;'>🍌 Fresh & Ready!</h2>", unsafe_allow_html=True)
        elif shelf_life > 2:
            st.markdown("<h2 style='text-align: center; color: #f39c12;'>🍌 Perfectly Ripe!</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='text-align: center; color: #e74c3c;'>🥀 Use it for Banana Bread!</h2>", unsafe_allow_html=True)
