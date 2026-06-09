import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta, timezone
import requests
import concurrent.futures
import time
import os
from dotenv import load_dotenv

# ==========================================
# CONFIGURATION & SECURITY
# ==========================================

load_dotenv('api.env')


st.set_page_config(
    page_title="AQI Forecaster", 
    page_icon="", 
    layout="wide", 
    initial_sidebar_state="expanded"
)


API_KEY = os.getenv('OPENAQ_API_KEY') or st.secrets.get("OPENAQ_API_KEY")

CITIES = {
    'Delhi': {'lat': 28.6139, 'lon': 77.2090}, 'Mumbai': {'lat': 19.0760, 'lon': 72.8777},
    'Chennai': {'lat': 13.0827, 'lon': 80.2707}, 'Kolkata': {'lat': 22.5726, 'lon': 88.3639},
    'Bangalore': {'lat': 12.9716, 'lon': 77.5946}, 'Hyderabad': {'lat': 17.3850, 'lon': 78.4867},
    'Pune': {'lat': 18.5204, 'lon': 73.8567}, 'Ahmedabad': {'lat': 23.0225, 'lon': 72.5714},
    'Jaipur': {'lat': 26.9124, 'lon': 75.7873}, 'Lucknow': {'lat': 26.8467, 'lon': 80.9462},
}

EXPECTED_POLLUTANTS = ['pm25', 'pm10', 'no', 'no2', 'nh3', 'co', 'so2', 'o3']

BREAKPOINTS = {
    'PM2.5': [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200), (91, 120, 201, 300), (121, 250, 301, 400), (251, 500, 401, 500)],
    'PM10': [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200), (251, 350, 201, 300), (351, 430, 301, 400), (431, 500, 401, 500)],
    'NO2': [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200), (181, 280, 201, 300), (281, 400, 301, 400), (401, 500, 401, 500)],
    'SO2': [(0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200), (381, 800, 201, 300), (801, 1600, 301, 400), (1601, 2100, 401, 500)],
    'CO': [(0, 1.0, 0, 50), (1.1, 2.0, 51, 100), (2.1, 10.0, 101, 200), (10.1, 17.0, 201, 300), (17.1, 34.0, 301, 400), (34.1, 50.0, 401, 500)],
    'O3': [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200), (169, 208, 201, 300), (209, 748, 301, 400), (749, 900, 401, 500)],
    'NH3': [(0, 200, 0, 50), (201, 400, 51, 100), (401, 800, 101, 200), (801, 1200, 201, 300), (1201, 1800, 301, 400), (1801, 2400, 401, 500)],
}

# ==========================================
# CPCB MATH FUNCTIONS
# ==========================================
def get_aqi_subindex(C, breakpoints):
    if C is None or pd.isna(C) or C < 0:
        return None
    for (C_low, C_high, AQI_low, AQI_high) in breakpoints:
        if C_low <= C <= C_high:
            AQI = ((AQI_high - AQI_low) / (C_high - C_low)) * (C - C_low) + AQI_low
            return round(AQI)
    return None

def calculate_aqi(row):
    sub_indices = []
    for pollutant, bp_table in BREAKPOINTS.items():
        value = row.get(pollutant) 
        sub_index = get_aqi_subindex(value, bp_table)
        if sub_index is not None:
            sub_indices.append(sub_index)
    if sub_indices:
        return max(sub_indices)
    return 0 

# ==========================================
# BACKEND ARCHITECTURE 
# ==========================================
def generate_fallback_data(city_name):
    """Generates 8 days of highly realistic dummy data if sensors are offline."""
    dates = [datetime.now(timezone.utc).date() - timedelta(days=x) for x in range(8, -1, -1)]
    base_pm25 = np.random.uniform(50, 90)
    records = []
    
    for d in dates:
        base_pm25 += np.random.uniform(-8, 8)
        base_pm25 = max(15, base_pm25) 
        records.append({
            'Date': d, 'PM2.5': base_pm25, 'PM10': base_pm25 * 1.8 + np.random.uniform(-10, 10), 
            'NO': np.random.uniform(10, 30), 'NO2': np.random.uniform(20, 50),
            'NH3': np.random.uniform(5, 15), 'CO': np.random.uniform(0.8, 1.8) * 1000, 
            'SO2': np.random.uniform(10, 30), 'O3': np.random.uniform(20, 60)
        })
    return pd.DataFrame(records).set_index('Date')

@st.cache_data(ttl=3600)
def fetch_live_city_data(city_name):
    """Fetches 8 days of live API data with API Rate Limiting (Throttling)."""
    headers = {"X-API-Key": API_KEY}
    coords = CITIES[city_name]
    
    date_to = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=8) 
    
    fmt_date_to = date_to.strftime("%Y-%m-%dT%H:%M:%S%z")
    fmt_date_from = date_from.strftime("%Y-%m-%dT%H:%M:%S%z")
    
    loc_url = "https://api.openaq.org/v3/locations"
    loc_params = {"coordinates": f"{coords['lat']},{coords['lon']}", "radius": 25000, "limit": 100}
    
    try:
        loc_resp = requests.get(loc_url, headers=headers, params=loc_params)
        
        # Graceful 429 Handling
        if loc_resp.status_code == 429:
            st.sidebar.warning("OpenAQ Rate Limit Hit. Defaulting to AI simulation.")
            return None, False
        elif loc_resp.status_code != 200:
            st.sidebar.error(f"OpenAQ Error {loc_resp.status_code}: {loc_resp.text}")
            return None, False

        locations = loc_resp.json().get('results', [])[:7] 
        
        fetch_tasks = []
        for loc in locations:
            for sensor in loc.get('sensors', []):
                parameter = sensor.get('parameter', {}).get('name')
                if parameter in EXPECTED_POLLUTANTS:
                    meas_url = f"https://api.openaq.org/v3/sensors/{sensor['id']}/days"
                    meas_params = {"datetime_from": fmt_date_from, "datetime_to": fmt_date_to, "limit": 100}
                    fetch_tasks.append((parameter, meas_url, meas_params))

        city_records = []
        def fetch_sensor_data(task):
            param, url, params = task
            time.sleep(0.3)
            resp = requests.get(url, headers=headers, params=params)
            if resp.status_code == 200:
                return param, resp.json().get('results', [])
            return param, []

        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = executor.map(fetch_sensor_data, fetch_tasks)

        for parameter, rows in results:
            for row in rows:
                city_records.append({
                    'Date': pd.to_datetime(row['period']['datetimeFrom']['utc']).date(),
                    'Parameter': parameter, 'Value': row.get('value', None)
                })
                        
        if not city_records: return None, False
            
        df_temp = pd.DataFrame(city_records)
        df_daily = df_temp.groupby(['Date', 'Parameter'])['Value'].mean().reset_index()
        df_pivot = df_daily.pivot(index='Date', columns='Parameter', values='Value')
        df_pivot = df_pivot.reindex(columns=EXPECTED_POLLUTANTS).ffill().bfill().fillna(0)
        df_pivot.rename(columns={'pm25': 'PM2.5', 'pm10': 'PM10', 'no': 'NO', 'no2': 'NO2', 'nh3': 'NH3', 'co': 'CO', 'so2': 'SO2', 'o3': 'O3'}, inplace=True)
        df_pivot['CO'] = df_pivot['CO'].clip(lower=0, upper=10000)
        # Flatline Detector
        recent_std = df_pivot['PM2.5'].tail(7).std()
        if recent_std == 0.0 or pd.isna(recent_std):
            return None, False

        return df_pivot.sort_index(), True
        
    except Exception as e:
        return None, False

def extract_model_features(df_live, city_name):
    df_calc = df_live.copy()
    df_calc['CO'] = df_calc['CO'] / 1000
    df_calc['AQI_calculated'] = df_calc.apply(calculate_aqi, axis=1)
    
    today_data = df_live.iloc[-1].to_dict()
    today = datetime.now(timezone.utc).date()
    month = today.month
    
    if month in [12, 1, 2]: season = 'Winter'
    elif month in [3, 4, 5]: season = 'Spring'
    elif month in [6, 7, 8, 9]: season = 'Monsoon'
    else: season = 'Post-Monsoon'

    return pd.DataFrame({
        'City': [city_name],
        'PM2.5': [today_data.get('PM2.5', 0)],
        'PM10': [today_data.get('PM10', 0)],
        'NO': [today_data.get('NO', 0)],
        'NO2': [today_data.get('NO2', 0)],
        'NH3': [today_data.get('NH3', 0)],
        'CO': [today_data.get('CO', 0)],
        'SO2': [today_data.get('SO2', 0)],
        'O3': [today_data.get('O3', 0)],
        'Month': [month],
        'Seasons': [season],
        'AQI_calculated': [df_calc['AQI_calculated'].iloc[-1]],
        'AQI_lag1': [df_calc['AQI_calculated'].iloc[-2] if len(df_calc) >= 2 else 0],
        'AQI_lag3': [df_calc['AQI_calculated'].iloc[-4] if len(df_calc) >= 4 else 0],
        'AQI_lag7': [df_calc['AQI_calculated'].iloc[-8] if len(df_calc) >= 8 else (df_calc['AQI_calculated'].iloc[0] if len(df_calc) > 0 else 0)],
        'DayOfWeek': [today.weekday()],
        'rolling_mean_7': [df_calc['AQI_calculated'].iloc[-7:].mean() if len(df_calc) >= 7 else df_calc['AQI_calculated'].iloc[-1]]
    })

# ==========================================
# STREAMLIT USER INTERFACE
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load('aqi_pipeline_v2.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load the model file 'aqi_pipeline_v2.pkl'. Ensure it is in the same directory. Error: {e}")
    st.stop()

st.title("Real-Time Air Quality AI Forecaster")
st.markdown("Predicting tomorrow's atmospheric conditions utilizing XGBoost and OpenAQ physical telemetry.")


if API_KEY:
    st.sidebar.success(f"Secure API Pipeline Active")
else:
    st.sidebar.error("API Key is missing! Check your api.env file.")

st.sidebar.header("Control Panel")
selected_city = st.sidebar.selectbox("Target City", list(CITIES.keys()))

if st.sidebar.button("Fetch Live Data & Predict", type="primary", use_container_width=True):
    with st.spinner(f"Connecting to ground sensors in {selected_city}..."):
        
        # 1. Fetch Data
        live_data, is_real_data = fetch_live_city_data(selected_city)
        
        if not is_real_data:
            st.warning(f"**Physical Sensors Offline or Throttled:** Engaging AI Simulation Mode.")
            live_data = generate_fallback_data(selected_city)
        else:
            st.success(f"Live telemetry secured from OpenAQ sensors in {selected_city}.")

        
        model_input = extract_model_features(live_data, selected_city)
        current_aqi = int(model_input['AQI_calculated'].iloc[0])
        
    
        raw_prediction = model.predict(model_input)[0]
        final_prediction = int(max(0, raw_prediction)) 
        
        
        if final_prediction <= 50: 
            color, category = "🟢", "Good"
        elif final_prediction <= 100: 
            color, category = "🟡", "Satisfactory"
        elif final_prediction <= 200: 
            color, category = "🟠", "Moderate"
        elif final_prediction <= 300: 
            color, category = "🔴", "Poor"
        elif final_prediction <= 400: 
            color, category = "🟣", "Very Poor"
        else: 
            color, category = "🟤", "Severe"

       # 5. Display Results
        st.markdown("---")
        st.subheader(f"Atmospheric Forecast: {selected_city}")
        st.write(f"**Season:** {model_input['Seasons'].iloc[0]} | **Baseline Calculated AQI Today:** {current_aqi}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Predicted AQI (Tomorrow)", value=f"{final_prediction} AQI", delta=f"{final_prediction - current_aqi} points", delta_color="inverse")
        with col2:
            st.markdown(f"### {color} **{category}**")
            if final_prediction > 200:
                st.error("High pollution warning in effect for tomorrow.")
            else:
                st.info("Air quality is expected to remain within safe thresholds.")
                
        st.markdown("---")
        st.write("Latest Sensor Readings (µg/m³)")
        
        display_df = live_data.tail(1).reset_index(drop=True)
        display_df.index = ["Telemetry"] 
        
        st.dataframe(display_df, use_container_width=True)