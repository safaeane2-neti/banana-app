# app.py - The complete Twinsie System for Streamlit!

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ============================================================================
# PASTE ALL YOUR CLASSES AND FUNCTIONS FROM YOUR COLAB SCRIPT HERE
# ============================================================================

# --- Data Generator (from your script) ---
class BananaRipeningSimulator:
    def __init__(self):
        self.R = 8.314
        self.Ea = 50000
        self.k0 = 0.35
        self.T_optimal = 20
        self.RH_optimal = 87.5
    def temperature_factor(self, temp_celsius):
        factor = np.exp(0.08 * (temp_celsius - self.T_optimal))
        return factor
    def humidity_factor(self, humidity):
        deviation = abs(humidity - self.RH_optimal)
        factor = 1 - (deviation / 100)
        return np.clip(factor, 0.3, 1.0)
    def ethylene_factor(self, ethylene_ppm):
        factor = 1 + (ethylene_ppm / 50)
        return np.clip(factor, 0.8, 2.5)
    def ripeness_ode(self, ripeness, t, temp, humidity, ethylene):
        R_max = 7.0
        k_temp = self.temperature_factor(temp)
        k_hum = self.humidity_factor(humidity)
        k_eth = self.ethylene_factor(ethylene)
        k_total = self.k0 * k_temp * k_hum * k_eth
        sigmoid = (1 - ripeness / R_max) if ripeness < R_max else 0
        dR_dt = k_total * sigmoid
        return dR_dt

# --- Economic Engine (from your script) ---
class BananaEconomics:
    def __init__(self):
        self.market_prices = {'export': 2.8, 'wholesale': 2.2, 'retail': 3.5, 'discount': 2.0, 'processing': 0.9}
        self.daily_storage_cost = 0.08
        self.transport_cost = 0.15
    def calculate_ripeness(self, shelf_life, current_day):
        total_life = current_day + shelf_life
        ripeness = 1 + (current_day / total_life) * 6
        return min(7, max(1, ripeness))
    def get_market_segment(self, ripeness):
        if ripeness < 3: return 'Export', self.market_prices['export'], '🌍'
        elif ripeness < 4: return 'Wholesale', self.market_prices['wholesale'], '🏪'
        elif ripeness < 5.5: return 'Retail', self.market_prices['retail'], '🛒'
        elif ripeness < 6.5: return 'Discount', self.market_prices['discount'], '💸'
        else: return 'Processing', self.market_prices['processing'], '🍞'
    def calculate_value(self, ripeness, days_stored, quality=0.95):
        market, base_price, emoji = self.get_market_segment(ripeness)
        price = base_price * quality
        cost = (self.daily_storage_cost * days_stored) + self.transport_cost
        net_value = max(0, price - cost)
        return {'market': market, 'emoji': emoji, 'base_price': base_price, 'net_value': net_value}
    def find_optimal_day(self, temp, humidity, ethylene):
        best_day, best_value = 1, 0
        for day in range(1, 11):
            shelf_life = predict_shelf_life(temp, humidity, ethylene, day)
            ripeness = self.calculate_ripeness(shelf_life, day)
            value_info = self.calculate_value(ripeness, day)
            if value_info['net_value'] > best_value:
                best_value = value_info['net_value']
                best_day = day
        return best_day, best_value

# --- Feature Engineering (from your script) ---
def engineer_features(df):
    df = df.copy()
    df['day_squared'] = df['day'] ** 2
    df['day_cubed'] = df['day'] ** 3
    df['temp_humidity_interaction'] = df['temperature_celsius'] * df['humidity_percent']
    df['temp_ethylene_interaction'] = df['temperature_celsius'] * df['ethylene_ppm']
    df['temp_normalized'] = (df['temperature_celsius'] - 20) / 10
    df['humidity_normalized'] = (df['humidity_percent'] - 75) / 25
    df['arrhenius_factor'] = np.exp(0.08 * (df['temperature_celsius'] - 20))
    df['humidity_stress'] = np.abs(df['humidity_percent'] - 87.5) / 50.0
    return df

# --- Prediction API (from your script) ---
def predict_shelf_life(temp, humidity, ethylene, day):
    input_df = pd.DataFrame([{'temperature_celsius': temp, 'humidity_percent': humidity, 'ethylene_ppm': ethylene, 'day': day}])
    input_df = engineer_features(input_df)
    input_processed = input_df[feature_cols]
    input_scaled = scaler.transform(input_processed)
    prediction = model.predict(input_scaled)[0]
    return np.clip(prediction, 0, 14)

# --- Banana Character (from your script) ---
def get_banana_character(ripeness):
    if ripeness < 2.5: return '🍌✨', 'Fresh & Happy!', '#c8e6c9'
    elif ripeness < 4.5: return '🍌😊', 'Perfect & Ready!', '#fff9c4'
    elif ripeness < 6: return '🍌😟', 'Getting Ripe!', '#ffe0b2'
    else: return '🍌💔', 'Overripe Alert!', '#ffccbc'

# --- HTML Dashboard (from your script) ---
def create_dashboard_html(temp, humidity, ethylene, day, shelf_life, economics_info, optimal_info):
    ripeness = economics.calculate_ripeness(shelf_life, day)
    banana_emoji, status_text, bg_color = get_banana_character(ripeness)
    # (Pasting the giant HTML string here. It's long but it's just one variable)
    html = f"""
    <div style="font-family: 'Comic Sans MS', 'Arial', sans-serif; max-width: 1200px; margin: 20px auto;">
        <div style="background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%); padding: 30px; border-radius: 20px; text-align: center; box-shadow: 0 10px 25px rgba(0,0,0,0.15); margin-bottom: 20px;">
            <h1 style="color: #2d3436; font-size: 56px; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);">🍌 Twinsie Dashboard 🍌</h1>
            <p style="color: #636e72; font-size: 20px; margin: 10px 0 0 0;">your personal ai-powered banana predictor bestie</p>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
            <div style="background: {bg_color}; padding: 30px; border-radius: 20px; box-shadow: 0 10px 25px rgba(0,0,0,0.15); text-align: center;">
                <div style="font-size: 140px; margin: 20px 0; animation: bounce 2s infinite;">{banana_emoji}</div>
                <h2 style="color: #2d3436; font-size: 36px; margin: 10px 0; font-weight: bold;">{status_text}</h2>
                <div style="background: white; border-radius: 15px; padding: 20px; margin-top: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.1);">
                    <p style="font-size: 20px; color: #636e72; margin: 8px 0;"><strong>Ripeness:</strong> <span style="color: #e74c3c; font-size: 24px;">{ripeness:.1f}/7</span></p>
                    <p style="font-size: 20px; color: #636e72; margin: 8px 0;"><strong>Current Day:</strong> <span style="color: #3498db; font-size: 24px;">Day {day:.1f}</span></p>
                </div>
            </div>
            <div style="background: white; padding: 30px; border-radius: 20px; box-shadow: 0 10px 25px rgba(0,0,0,0.15);">
                <h3 style="color: #2d3436; font-size: 28px; margin-top: 0; border-bottom: 3px solid #fdcb6e; padding-bottom: 10px;">📊 Live Sensor Data</h3>
                <div style="margin: 20px 0; padding: 18px; background: linear-gradient(135deg, #fff5f5 0%, #ffe0e0 100%); border-radius: 12px; border-left: 5px solid #e74c3c;"><span style="font-size: 28px;">🌡️</span><strong style="color: #e74c3c; font-size: 18px;">Temperature:</strong><span style="font-size: 24px; color: #2d3436; font-weight: bold;">{temp:.1f}°C</span></div>
                <div style="margin: 20px 0; padding: 18px; background: linear-gradient(135deg, #f0f8ff 0%, #d6eaf8 100%); border-radius: 12px; border-left: 5px solid #3498db;"><span style="font-size: 28px;">💧</span><strong style="color: #3498db; font-size: 18px;">Humidity:</strong><span style="font-size: 24px; color: #2d3436; font-weight: bold;">{humidity:.1f}%</span></div>
                <div style="margin: 20px 0; padding: 18px; background: linear-gradient(135deg, #f0fff4 0%, #d4edda 100%); border-radius: 12px; border-left: 5px solid #27ae60;"><span style="font-size: 28px;">🌿</span><strong style="color: #27ae60; font-size: 18px;">Ethylene:</strong><span style="font-size: 24px; color: #2d3436; font-weight: bold;">{ethylene:.1f} ppm</span></div>
                <div style="margin: 20px 0; padding: 18px; background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); border-radius: 12px; border-left: 5px solid #f39c12;"><span style="font-size: 28px;">⏳</span><strong style="color: #f39c12; font-size: 18px;">Shelf Life:</strong><span style="font-size: 28px; color: #2d3436; font-weight: bold;">{shelf_life:.1f} days</span></div>
            </div>
        </div>
        <div style="background: linear-gradient(135deg, #a8e6cf 0%, #dcedc1 100%); padding: 30px; border-radius: 20px; box-shadow: 0 10px 25px rgba(0,0,0,0.15); margin-bottom: 20px;">
            <h3 style="color: #2d3436; font-size: 32px; text-align: center; margin-top: 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.1);">💰 Economic Optimization Engine</h3>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-top: 25px;">
                <div style="background: white; padding: 25px; border-radius: 18px; text-align: center; box-shadow: 0 6px 15px rgba(0,0,0,0.1);"><div style="font-size: 50px; margin-bottom: 10px;">{economics_info['emoji']}</div><p style="font-size: 14px; color: #95a5a6; margin: 5px 0; text-transform: uppercase; letter-spacing: 1px;">Current Market</p><p style="font-size: 24px; color: #2d3436; font-weight: bold; margin: 10px 0;">{economics_info['market']}</p><p style="font-size: 20px; color: #27ae60; font-weight: bold; margin: 5px 0;">${economics_info['base_price']:.2f}/kg</p></div>
                <div style="background: white; padding: 25px; border-radius: 18px; text-align: center; box-shadow: 0 6px 15px rgba(0,0,0,0.1);"><div style="font-size: 50px; margin-bottom: 10px;">💵</div><p style="font-size: 14px; color: #95a5a6; margin: 5px 0; text-transform: uppercase; letter-spacing: 1px;">Net Value (Day {day:.0f})</p><p style="font-size: 32px; color: #27ae60; font-weight: bold; margin: 10px 0;">${economics_info['net_value']:.2f}</p><p style="font-size: 16px; color: #7f8c8d; margin: 5px 0;">per kilogram</p></div>
                <div style="background: white; padding: 25px; border-radius: 18px; text-align: center; box-shadow: 0 6px 15px rgba(0,0,0,0.1);"><div style="font-size: 50px; margin-bottom: 10px;">🎯</div><p style="font-size: 14px; color: #95a5a6; margin: 5px 0; text-transform: uppercase; letter-spacing: 1px;">Optimal Ship Day</p><p style="font-size: 32px; color: #e74c3c; font-weight: bold; margin: 10px 0;">Day {optimal_info['day']}</p><p style="font-size: 16px; color: #27ae60; font-weight: bold; margin: 5px 0;">Max: ${optimal_info['value']:.2f}/kg</p></div>
            </div>
            <div style="background: rgba(255,255,255,0.95); padding: 25px; border-radius: 18px; margin-top: 20px; text-align: center; border: 4px solid #27ae60; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                <p style="font-size: 20px; color: #2d3436; margin: 8px 0; font-weight: bold;">📦 Per Container Projection (18,000 kg)</p>
                <p style="font-size: 36px; color: #27ae60; font-weight: bold; margin: 15px 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.1);">${economics_info['net_value'] * 18000:,.0f}</p>
                <p style="font-size: 16px; color: #636e72; margin: 8px 0;">💡 Potential gain with optimization: <strong style="color: #e74c3c; font-size: 20px;">${(optimal_info['value'] - economics_info['net_value']) * 18000:,.0f}</strong></p>
            </div>
        </div>
        <div style="background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%); padding: 25px; border-radius: 18px; box-shadow: 0 10px 25px rgba(0,0,0,0.15); text-align: center;">
            <p style="font-size: 22px; color: #2d3436; margin: 0; font-weight: bold;">{"💡 ✨ Ship now for best profit!" if abs(day - optimal_info['day']) < 0.5 else f"💡 ⏰ Wait {optimal_info['day'] - day:.0f} more days for optimal profit!" if day < optimal_info['day'] else "💡 ⚠️ You've passed the optimal shipping window!"}</p>
        </div>
    </div><style>@keyframes bounce {{0%, 100% {{ transform: translateY(0); }} 50% {{ transform: translateY(-10px); }}</style>"""
    return st.components.v1.html(html, height=1400)


# ============================================================================
# THE STREAMLIT APP ITSELF
# ============================================================================

# --- Load the model and artifacts ---
@st.cache_resource
def load_model():
    model = joblib.load('hybrid_model.pkl')
    scaler = joblib.load('scaler.pkl')
    with open('feature_cols.txt', 'r') as f:
        feature_cols = f.read().splitlines()
    return model, scaler, feature_cols

model, scaler, feature_cols = load_model()

# --- Page Setup ---
st.set_page_config(page_title="Twinsie Dashboard 🍌", page_icon="🍌", layout="centered")

st.title("🍌 Live Twinsie Dashboard 🍌")
st.markdown("Your personal AI-powered banana predictor bestie.")

# --- The Main App ---
if st.button("🎬 Start Live Simulation"):
    economics = BananaEconomics()
    for i in range(30):
        temp = 20 + np.random.uniform(-2, 5)
        humidity = 85 + np.random.uniform(-10, 10)
        ethylene = 15 + np.random.uniform(-5, 20)
        current_day = np.random.uniform(1, 8)
        
        shelf_life = predict_shelf_life(temp, humidity, ethylene, current_day)
        
        ripeness = economics.calculate_ripeness(shelf_life, current_day)
        economics_info = economics.calculate_value(ripeness, current_day)
        optimal_day, optimal_value = economics.find_optimal_day(temp, humidity, ethylene)
        optimal_info = {'day': optimal_day, 'value': optimal_value}
        
        create_dashboard_html(
            temp, humidity, ethylene, current_day, shelf_life,
            economics_info, optimal_info
        )
        
        my_placeholder = st.empty()
        my_placeholder.text(f"Next update in 5 seconds... ({i+1}/30)")
        time.sleep(5)
        my_placeholder.empty()
