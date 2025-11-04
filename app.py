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
# ALL CLASSES AND FUNCTIONS
# ============================================================================

# --- Economic Engine ---
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

# --- Feature Engineering ---
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

# --- Prediction API ---
def predict_shelf_life(temp, humidity, ethylene, day):
    input_df = pd.DataFrame([{'temperature_celsius': temp, 'humidity_percent': humidity, 'ethylene_ppm': ethylene, 'day': day}])
    input_df = engineer_features(input_df)
    input_processed = input_df[feature_cols]
    input_scaled = scaler.transform(input_processed)
    prediction = model.predict(input_scaled)[0]
    return np.clip(prediction, 0, 14)

# --- Clean & Minimalist HTML Dashboard ---
def create_dashboard_html(temp, humidity, ethylene, day, shelf_life, economics_info, optimal_info):
    
    # --- CUSTOMIZE YOUR MAIN COLOR HERE ---
    main_color = '#f39c12' # A nice, soft orange
    
    economics = BananaEconomics()
    ripeness = economics.calculate_ripeness(shelf_life, day)
    market, _, emoji = economics.get_market_segment(ripeness)

    html = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; max-width: 900px; margin: 40px auto; color: #333;">
        
        <!-- Header -->
        <div style="text-align: center; margin-bottom: 40px;">
            <h1 style="font-size: 42px; font-weight: 700; color: #333; margin: 0;">
                Twinsie Dashboard
            </h1>
            <p style="font-size: 18px; color: #777; margin-top: 8px;">
                AI-Powered Ripeness & Profitability Analysis
            </p>
        </div>
        
        <!-- Key Metrics -->
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px;">
            <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; border-left: 4px solid {main_color};">
                <div style="font-size: 24px; margin-bottom: 5px;">{emoji}</div>
                <div style="font-size: 14px; color: #777; text-transform: uppercase; letter-spacing: 0.5px;">Market</div>
                <div style="font-size: 20px; font-weight: 600; color: #333;">{market}</div>
            </div>
            <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center;">
                <div style="font-size: 24px; margin-bottom: 5px;">⏳</div>
                <div style="font-size: 14px; color: #777; text-transform: uppercase; letter-spacing: 0.5px;">Shelf Life</div>
                <div style="font-size: 20px; font-weight: 600; color: #333;">{shelf_life:.1f} days</div>
            </div>
            <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center;">
                <div style="font-size: 24px; margin-bottom: 5px;">💵</div>
                <div style="font-size: 14px; color: #777; text-transform: uppercase; letter-spacing: 0.5px;">Net Value</div>
                <div style="font-size: 20px; font-weight: 600; color: #333;">${economics_info['net_value']:.2f}/kg</div>
            </div>
            <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center;">
                <div style="font-size: 24px; margin-bottom: 5px;">🎯</div>
                <div style="font-size: 14px; color: #777; text-transform: uppercase; letter-spacing: 0.5px;">Optimal Day</div>
                <div style="font-size: 20px; font-weight: 600; color: #333;">Day {optimal_info['day']}</div>
            </div>
        </div>

        <!-- Details Section -->
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            
            <!-- Sensor Data -->
            <div style="background: #ffffff; padding: 25px; border-radius: 8px; border: 1px solid #e9ecef;">
                <h3 style="font-size: 18px; font-weight: 600; margin-top: 0; margin-bottom: 20px; color: #333;">Sensor Readings</h3>
                <div style="display: flex; justify-content: space-between; margin-bottom: 12px;"><span style="color: #777;">Temperature</span><span style="font-weight: 500;">{temp:.1f}°C</span></div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 12px;"><span style="color: #777;">Humidity</span><span style="font-weight: 500;">{humidity:.0f}%</span></div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 12px;"><span style="color: #777;">Ethylene</span><span style="font-weight: 500;">{ethylene:.1f} ppm</span></div>
                <div style="display: flex; justify-content: space-between;"><span style="color: #777;">Current Day</span><span style="font-weight: 500;">{day:.1f}</span></div>
            </div>
            
            <!-- Projection -->
            <div style="background: #ffffff; padding: 25px; border-radius: 8px; border: 1px solid #e9ecef;">
                <h3 style="font-size: 18px; font-weight: 600; margin-top: 0; margin-bottom: 20px; color: #333;">Container Projection</h3>
                <div style="text-align: center;">
                    <div style="font-size: 32px; font-weight: 700; color: {main_color};">
                        ${economics_info['net_value'] * 18000:,.0f}
                    </div>
                    <div style="font-size: 14px; color: #777; margin-top: 5px;">Current Value (18,000 kg)</div>
                </div>
                <hr style="border: none; border-top: 1px solid #e9ecef; margin: 20px 0;">
                <div style="text-align: center;">
                    <div style="font-size: 24px; font-weight: 600; color: #2ecc71;">
                        +${(optimal_info['value'] - economics_info['net_value']) * 18000:,.0f}
                    </div>
                    <div style="font-size: 14px; color: #777; margin-top: 5px;">Potential Gain with Optimization</div>
                </div>
            </div>
        </div>
        
        <!-- Recommendation -->
        <div style="background: {main_color}; color: white; padding: 20px; border-radius: 8px; text-align: center; margin-top: 20px;">
            <p style="font-size: 18px; font-weight: 600; margin: 0;">
                {"Ship now for optimal profit." if abs(day - optimal_info['day']) < 0.5 
                 else f"Wait {optimal_info['day'] - day:.0f} days for optimal profit." if day < optimal_info['day']
                 else "Optimal shipping window has passed."}
            </p>
        </div>

    </div>
    """
    
    return st.components.v1.html(html, height=700)


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
