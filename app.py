# app.py - Twinsie sie hooray
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
import warnings
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict

warnings.filterwarnings('ignore')
# ============================================================================
# ALL CLASSES & FUNCTIONS YIPPEE
# ============================================================================

# --- Model Card ---
MODEL_CARD = {
    "model_version": "1.2", "model_name": "Hybrid Physics-Informed Gradient Boosting Regressor",
    "training_data": "Synthetically generated data from a physics-informed simulator.",
    "trained_on": "2025-10-27", "metrics": {"rmse": 0.48, "mae": 0.32, "r_squared": 0.99},
    "limitations": ["Assumes a single banana cultivar.", "Not validated for long-distance sea transport."],
    "ethical_considerations": "This tool aims to optimize for both profit and equitable market access."
}

# --- Economic Engine🥀 ---
class AdaptiveEconomics:
    def __init__(self):
        self.base_market_prices = {'export': 2.8, 'wholesale': 2.2, 'retail': 3.5, 'discount': 2.0, 'processing': 0.9}
        self.market_prices = self.base_market_prices.copy()
        self.daily_storage_cost = 0.08
        self.transport_cost = 0.15
    def adjust_prices(self, supply_demand_ratio):
        adjustment = np.clip(1 + (0.2 * (1 - supply_demand_ratio)), 0.8, 1.2)
        for k in self.market_prices:
            self.market_prices[k] = self.base_market_prices[k] * adjustment
    def calculate_value(self, ripeness, days_stored, quality=0.95):
        market, base_price, emoji = self.get_market_segment(ripeness)
        price = base_price * quality
        cost = (self.daily_storage_cost * days_stored) + self.transport_cost
        net_value = max(0, price - cost)
        return {'market': market, 'emoji': emoji, 'base_price': base_price, 'net_value': net_value}
    def get_market_segment(self, ripeness):
        if ripeness < 3: return 'Export', self.market_prices['export'], '🌍'
        elif ripeness < 4: return 'Wholesale', self.market_prices['wholesale'], '🏪'
        elif ripeness < 5.5: return 'Retail', self.market_prices['retail'], '🛒'
        elif ripeness < 6.5: return 'Discount', self.market_prices['discount'], '💸'
        else: return 'Processing', self.market_prices['processing'], '🍞'
    def calculate_ripeness(self, shelf_life, current_day):
        total_life = current_day + shelf_life
        ripeness = 1 + (current_day / total_life) * 6
        return min(7, max(1, ripeness))
    def find_optimal_day(self, temp, humidity, ethylene):
        best_day, best_value = 1, 0
        for day in range(1, 11):
            shelf_life = predict_shelf_life(temp, humidity, ethylene, day)[0]
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
@st.cache_data
def predict_shelf_life(temp, humidity, ethylene, day):
    input_df = pd.DataFrame([{'temperature_celsius': temp, 'humidity_percent': humidity, 'ethylene_ppm': ethylene, 'day': day}])
    input_df = engineer_features(input_df)
    input_processed = input_df[feature_cols]
    input_scaled = scaler.transform(input_processed)
    prediction = model.predict(input_scaled)[0]
    tree_preds = np.array([tree.predict(input_scaled) for tree in model.estimators_])
    y_std = np.std(tree_preds)
    return np.clip(prediction, 0, 14), y_std

# --- Sustainability Metrics ---
def calculate_sustainability_impact(optimal_day, current_day, container_weight_kg=18000):
    waste_reduction_pct = calculate_waste_reduction(optimal_day, current_day)
    waste_saved_kg = (waste_reduction_pct / 100) * container_weight_kg * 0.1
    co2_saved_kg = waste_saved_kg * 2.5
    return waste_saved_kg, co2_saved_kg
def calculate_waste_reduction(optimal_day, current_day):
    days_late = max(0, current_day - optimal_day)
    potential_waste_per_day = 0.1
    total_potential_waste = days_late * potential_waste_per_day
    reduction = (1 - total_potential_waste) * 100
    return np.clip(reduction, 0, 100)

# ---Visual Banana: the star of the show yippee ---
def create_visual_banana(ripeness_score):
    x = np.linspace(0, 10, 100)
    y = 3 * np.sin(x / 2) + 5
    if ripeness_score < 0.5:
        color = f'rgb({int(255 * ripeness_score * 2)}, 255, 0)'
    else:
        brown_level = (ripeness_score - 0.5) * 2
        color = f'rgb(255, {int(255 * (1 - brown_level * 0.7))}, 0)'
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color=color, width=20), hoverinfo='none'))
    if ripeness_score > 0.6:
        num_spots = int((ripeness_score - 0.6) * 25)
        spot_x = np.random.uniform(1, 9, num_spots)
        spot_y = np.random.uniform(4, 6, num_spots)
        fig.add_trace(go.Scatter(x=spot_x, y=spot_y, mode='markers', marker=dict(color='saddlebrown', size=6), hoverinfo='none'))
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x", scaleratio=1), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=0, b=0), height=200, showlegend=False)
    return fig

# --- The Hybrid Model Showcase ---
def create_model_comparison_chart(temp, humidity, ethylene):
    economics = AdaptiveEconomics()
    data = []
    for day in range(1, 11):
        hybrid_shelf_life = predict_shelf_life(temp, humidity, ethylene, day)[0]
        hybrid_ripeness = economics.calculate_ripeness(hybrid_shelf_life, day)
        hybrid_value = economics.calculate_value(hybrid_ripeness, day)['net_value']
        physics_shelf_life = 7.0 - (0.35 * np.exp(0.08 * (temp - 20)) * (1 + ethylene / 50) * day)
        physics_ripeness = economics.calculate_ripeness(physics_shelf_life, day)
        physics_value = economics.calculate_value(physics_ripeness, day)['net_value']
        data.append({'Day': day, 'Hybrid Model ($)': hybrid_value, 'Physics Model ($)': physics_value})
    df_chart = pd.DataFrame(data)
    fig = px.line(df_chart, x='Day', y=['Hybrid Model ($)', 'Physics Model ($)'], title='Hybrid vs. Pure Physics Model', markers=True)
    fig.update_layout(template="simple_white")
    return fig

# --- Feedback Loop Stub ---
def update_model_with_feedback(observed_shelf_life, last_input):
    st.info(" Feedback received. Model update queued for next training cycle.")
    return True

# --- Dashboard HTML ---
def create_dashboard_html(temp, humidity, ethylene, day, shelf_life, y_std, economics_info, optimal_info, waste_reduction, co2_saved):
    main_color = '#f39c12'
    economics = AdaptiveEconomics()
    ripeness = economics.calculate_ripeness(shelf_life, day)
    market, _, emoji = economics.get_market_segment(ripeness)
    html = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; max-width: 900px; margin: 40px auto; color: #333;">
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-bottom: 30px;">
            <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; border-left: 4px solid {main_color};">
                <div style="font-size: 24px; margin-bottom: 5px;">{emoji}</div>
                <div style="font-size: 14px; color: #777; text-transform: uppercase; letter-spacing: 0.5px;">Market</div>
                <div style="font-size: 20px; font-weight: 600; color: #333;">{market}</div>
            </div>
            <div style="background: #e8f5e9; padding: 20px; border-radius: 8px; text-align: center; border-left: 4px solid #2ecc71;">
                <div style="font-size: 24px; margin-bottom: 5px;">🌱</div>
                <div style="font-size: 14px; color: #777; text-transform: uppercase; letter-spacing: 0.5px;">Waste Reduction</div>
                <div style="font-size: 20px; font-weight: 600; color: #333;">{waste_reduction:.1f}%</div>
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 30px;">
            <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center;">
                <div style="font-size: 24px; margin-bottom: 5px;">⏳</div>
                <div style="font-size: 14px; color: #777; text-transform: uppercase; letter-spacing: 0.5px;">Shelf Life</div>
                <div style="font-size: 20px; font-weight: 600; color: #333;">{shelf_life:.1f} ± {y_std:.2f} days</div>
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
        <div style="background: #e8f5e9; padding: 20px; border-radius: 8px; text-align: center; margin-bottom: 30px; border: 1px solid #c3e6cb;">
            <h3 style="margin-top:0;">🌍 Sustainability Impact</h3>
            <p style="font-size: 16px;">Optimizing this shipment saves an estimated <strong>{co2_saved:.0f} kg of CO₂e</strong>.</p>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div style="background: #ffffff; padding: 25px; border-radius: 8px; border: 1px solid #e9ecef;">
                <h3 style="font-size: 18px; font-weight: 600; margin-top: 0; margin-bottom: 20px; color: #333;">Sensor Readings</h3>
                <div style="display: flex; justify-content: space-between; margin-bottom: 12px;"><span style="color: #777;">Temperature</span><span style="font-weight: 500;">{temp:.1f}°C</span></div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 12px;"><span style="color: #777;">Humidity</span><span style="font-weight: 500;">{humidity:.0f}%</span></div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 12px;"><span style="color: #777;">Ethylene</span><span style="font-weight: 500;">{ethylene:.1f} ppm</span></div>
                <div style="display: flex; justify-content: space-between;"><span style="color: #777;">Current Day</span><span style="font-weight: 500;">{day:.1f}</span></div>
            </div>
            <div style="background: #ffffff; padding: 25px; border-radius: 8px; border: 1px solid #e9ecef;">
                <h3 style="font-size: 18px; font-weight: 600; margin-top: 0; margin-bottom: 20px; color: #333;">Container Projection</h3>
                <div style="text-align: center;">
                    <div style="font-size: 32px; font-weight: 700; color: {main_color};">${economics_info['net_value'] * 18000:,.0f}</div>
                    <div style="font-size: 14px; color: #777; margin-top: 5px;">Current Value (18,000 kg)</div>
                </div>
                <hr style="border: none; border-top: 1px solid #e9ecef; margin: 20px 0;">
                <div style="text-align: center;">
                    <div style="font-size: 24px; font-weight: 600; color: #2ecc71;">+${(optimal_info['value'] - economics_info['net_value']) * 18000:,.0f}</div>
                    <div style="font-size: 14px; color: #777; margin-top: 5px;">Potential Gain with Optimization</div>
                </div>
            </div>
        </div>
        <div style="background: {main_color}; color: white; padding: 20px; border-radius: 8px; text-align: center; margin-top: 20px;">
            <p style="font-size: 18px; font-weight: 600; margin: 0;">{"Ship now for optimal profit." if abs(day - optimal_info['day']) < 0.5 else f"Wait {optimal_info['day'] - day:.0f} days for optimal profit." if day < optimal_info['day'] else "Optimal shipping window has passed."}</p>
        </div>
    </div>
    """
    return st.components.v1.html(html, height=800)
# ============================================================================
# THE STREAMLIT APP
# ============================================================================
@st.cache_resource
def load_model():
    model = joblib.load('hybrid_model.pkl')
    scaler = joblib.load('scaler.pkl')
    with open('feature_cols.txt', 'r') as f:
        feature_cols = f.read().splitlines()
    return model, scaler, feature_cols

model, scaler, feature_cols = load_model()
st.set_page_config(page_title="Twinsie Dashboard 🍌", page_icon="🍌", layout="centered")

# --- Sidebar ---
st.sidebar.header("🔧 Simulation Controls")
temp = st.sidebar.slider("🌡️ Temperature (°C)", 15.0, 30.0, 22.0)
humidity = st.sidebar.slider("💧 Humidity (%)", 60.0, 95.0, 85.0)
ethylene = st.sidebar.slider("🍈 Ethylene (ppm)", 5.0, 50.0, 20.0)
current_day = st.sidebar.slider("📅 Current Day", 0.5, 12.0, 3.0)
st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Intelligent Market Simulation")
supply_demand_ratio = st.sidebar.slider("Supply vs. Demand", 0.5, 1.5, 1.0)
run_simulation = st.sidebar.toggle("▶️ Run Live Simulation")

# --- Main App Content ---
st.title("🍌 Live Twinsie Dashboard 🍌")
st.markdown("Your personal AI-powered banana predictor bestie.")

# --- Initial Analysis ---
economics = AdaptiveEconomics()
economics.adjust_prices(supply_demand_ratio)
shelf_life, y_std = predict_shelf_life(temp, humidity, ethylene, current_day)
ripeness = economics.calculate_ripeness(shelf_life, current_day)
economics_info = economics.calculate_value(ripeness, current_day)
optimal_day, optimal_value = economics.find_optimal_day(temp, humidity, ethylene)
optimal_info = {'day': optimal_day, 'value': optimal_value}
waste_reduction = calculate_waste_reduction(optimal_day, current_day)
waste_saved_kg, co2_saved_kg = calculate_sustainability_impact(optimal_day, current_day)

# --- The Visual Twin  ---
st.subheader("🍌 Your Digital Twin")
visual_banana_fig = create_visual_banana(ripeness)
st.plotly_chart(visual_banana_fig, use_container_width=True)

# --- The Hybrid Model Showcase ---
st.subheader("🧠 Why Our Hybrid Model is Superior")
model_comparison_fig = create_model_comparison_chart(temp, humidity, ethylene)
st.plotly_chart(model_comparison_fig, use_container_width=True)

# --- metrics and info ---
st.markdown("---")
create_dashboard_html(
    temp, humidity, ethylene, current_day, shelf_life, y_std,
    economics_info, optimal_info, waste_reduction, co2_saved_kg
)

# --- Live Simulation ---
if run_simulation:
    placeholder = st.empty()
    for i in range(30):
        sim_temp = temp + np.random.uniform(-1, 1)
        sim_humidity = humidity + np.random.uniform(-5, 5)
        sim_ethylene = ethylene + np.random.uniform(-2, 2)
        sim_shelf_life, _ = predict_shelf_life(sim_temp, sim_humidity, sim_ethylene, current_day)
        with placeholder.container():
            st.info(f"🔄 LIVE UPDATE {i+1}/30")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Temp", f"{sim_temp:.1f}°C")
            col2.metric("Humidity", f"{sim_humidity:.0f}%")
            col3.metric("Ethylene", f"{sim_ethylene:.1f} ppm")
            col4.metric("Shelf Life", f"{sim_shelf_life:.1f} days")
        time.sleep(5)
    placeholder.success("Simulation complete😆")
    time.sleep(2)
    placeholder.empty()
    st.rerun()
