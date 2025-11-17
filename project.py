import streamlit as st
import pandas as pd
import numpy as np
import json
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import folium
from streamlit_folium import st_folium
from branca.colormap import LinearColormap
import plotly.graph_objects as go

# ======================================================================================
# Page Configuration
# ======================================================================================
st.set_page_config(page_title="Water Quality Forecast Dashboard", layout="wide")

# ======================================================================================
# Data Processing Function
# ======================================================================================
def process_data(uploaded_file):
    """Loads, cleans, and pre-processes the user-uploaded CSV file."""
    try:
        df = pd.read_csv(uploaded_file)

        # --- Date and Coordinate Parsing ---
        def extract_date(system_index):
            if not isinstance(system_index, str): return pd.NaT
            parts = system_index.split('_')
            return pd.to_datetime(''.join(parts[:3]), format='%Y%m_%d', errors='coerce')

        def extract_coordinates(geo_json):
            try:
                geo_dict = json.loads(geo_json)
                coords = geo_dict.get('coordinates')
                if isinstance(coords, list) and len(coords) == 2:
                    return pd.Series(coords)
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass
            return pd.Series([np.nan, np.nan])

        df['date'] = df['system:index'].apply(extract_date)
        df[['longitude', 'latitude']] = df['.geo'].apply(extract_coordinates)

        df.dropna(subset=['date', 'longitude', 'latitude'], inplace=True)
        df.drop_duplicates(subset=['date', 'longitude', 'latitude'], keep='first', inplace=True)
        df.set_index('date', inplace=True)

        parameters = ['Turbidity', 'NDWI', 'Chlorophyll']
        for col in parameters:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].interpolate(method='time')

        df = df.bfill().ffill()
        df.reset_index(inplace=True)

        def cap_outliers(df, column, lower=0.05, upper=0.95):
            lower_bound = df[column].quantile(lower)
            upper_bound = df[column].quantile(upper)
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
            return df

        for col in parameters:
            df = cap_outliers(df, col)

        df_daily_agg = df.groupby('date').agg({
            'Turbidity': 'mean',
            'NDWI': 'mean',
            'Chlorophyll': 'mean'
        }).reset_index()

        df_for_map = df.copy()

        return df_daily_agg, df_for_map

    except Exception as e:
        st.error(f"An error occurred during data processing: {e}")
        st.warning("Please ensure your CSV has the correct format: system:index, .geo, Turbidity, NDWI, Chlorophyll.")
        return None, None

# ======================================================================================
# Landing Page
# ======================================================================================
def render_landing_page():
    st.title("🌊 Water Body Quality Forecast Dashboard")

    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; margin-bottom: 2rem;">
        <h3>Welcome! Start by uploading your data.</h3>
        <p>This tool uses the Prophet forecasting model to predict water quality parameters based on your time-series data.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader(
            "📂 Upload your pre-processed CSV file to begin",
            type=["csv"],
            help="The CSV must contain 'system:index', '.geo', 'Turbidity', 'NDWI', and 'Chlorophyll' columns."
        )

    if uploaded_file:
        with st.spinner("🔬 Processing your data... Please wait."):
            df_daily, df_for_map = process_data(uploaded_file)
            if df_daily is not None and df_for_map is not None:
                st.success("✅ Data processed successfully!")

                st.subheader("🔍 Preview of Processed Daily Data")
                st.dataframe(df_daily.head(10))

                csv = df_daily.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Processed CSV",
                    data=csv,
                    file_name='processed_daily_data.csv',
                    mime='text/csv'
                )

                st.session_state['processed_data'] = {
                    'df_daily': df_daily,
                    'df_for_map': df_for_map
                }
                st.rerun()

    st.markdown("""
    ---
    <div style="text-align: center;">
        <h4>Required CSV Format</h4>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.info("**system:index**\n\nMust be in YYYY_MM_DD_* format (e.g., 2023_05_15_A1B2).")
    with col_b:
        st.success("**.geo**\n\nMust be a GeoJSON string with coordinates (e.g., {\"type\":\"Point\",\"coordinates\":[lon, lat]}).")
    with col_c:
        st.warning("*Parameters*\n\nMust include numeric columns: Turbidity, NDWI, Chlorophyll.")

# ======================================================================================
# Dashboard Page
# ======================================================================================
def render_dashboard():
    df_daily = st.session_state['processed_data']['df_daily']
    df_for_map = st.session_state['processed_data']['df_for_map']
    parameters = ['Turbidity', 'NDWI', 'Chlorophyll']

    st.title("🌊 Water Quality Forecast Dashboard")

    st.sidebar.header("⚙ Controls")
    if st.sidebar.button("↩ Upload New File"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

    forecast_horizon = st.sidebar.slider(
        "Select Forecast Horizon (days)",
        min_value=30, max_value=730, value=365, step=30
    )

    st.header("📈 Time-Series Forecasts")
    forecast_results = {}

    for param in parameters:
        with st.container():
            st.subheader(f"📊 {param} Forecast")
            data = df_daily[['date', param]].rename(columns={'date': 'ds', param: 'y'}).dropna()

            if len(data) < 2:
                st.warning(f"⚠ Not enough data to generate a forecast for {param}.")
                continue

            model = Prophet(
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.1,
                seasonality_prior_scale=5.0,
                yearly_seasonality=True
            )
            model.fit(data)
            future = model.make_future_dataframe(periods=forecast_horizon, freq='D')
            forecast = model.predict(future)
            forecast_results[param] = forecast

            merged = pd.merge(data, forecast[['ds', 'yhat']], on='ds', how='inner')
            r2 = r2_score(merged['y'], merged['yhat'])
            rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))
            mae = mean_absolute_error(merged['y'], merged['yhat'])
            st.markdown(f"*Model Fit Metrics:* R²: {r2:.3f} | RMSE: {rmse:.3f} | MAE: {mae:.3f}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], mode='lines', name='Actual', line=dict(color='orange', width=2)))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='royalblue', width=2.5)))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', fill='tonexty', fillcolor='rgba(65,105,225,0.2)', line=dict(width=0), name='Confidence Interval'))
            fig.update_layout(title=f"{param} - Observed vs. Forecast", xaxis_title="Date", yaxis_title=param, legend=dict(orientation='h'), template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

    st.sidebar.header("📅 Select Prediction Date")
    min_date = df_daily['date'].max() + pd.Timedelta(days=1)
    max_date = min_date + pd.Timedelta(days=forecast_horizon - 1)
    selected_date = st.sidebar.date_input(
        "View predicted value on a specific date",
        min_value=min_date.date(), max_value=max_date.date(), value=min_date.date()
    )

    st.subheader(f"🔍 Predicted Values for {selected_date.strftime('%Y-%m-%d')}")
    cols = st.columns(len(parameters))
    for i, param in enumerate(parameters):
        if param in forecast_results:
            forecast = forecast_results[param]
            pred_row = forecast[forecast['ds'] == pd.to_datetime(selected_date)]
            if not pred_row.empty:
                pred_value = pred_row['yhat'].values[0]
                cols[i].metric(label=f"Predicted {param}", value=f"{pred_value:.3f}")
            else:
                cols[i].metric(label=f"Predicted {param}", value="N/A")

    st.header("🗺 Latest Spatial Heatmaps")
    st.info("These maps show the most recent measurement for each unique location.")

    for param in parameters:
        with st.container():
            st.subheader(f"📍 {param} Heatmap")
            df_param = df_for_map.dropna(subset=['latitude', 'longitude', param]).copy()

            if df_param.empty:
                st.warning(f"⚠ No spatial data for {param}.")
                continue

            latest_points = df_param.sort_values('date').drop_duplicates(subset=['latitude', 'longitude'], keep='last')
            avg_lat = latest_points['latitude'].mean()
            avg_lon = latest_points['longitude'].mean()
            m = folium.Map(location=[avg_lat, avg_lon], zoom_start=11, tiles='CartoDB positron')

            min_val, max_val = latest_points[param].min(), latest_points[param].max()
            if min_val == max_val:
                max_val += 1e-6

            colormap = LinearColormap(
                colors=['#2c7bb6', '#abd9e9', '#ffffbf', '#fdae61', '#d7191c'],
                vmin=min_val, vmax=max_val, caption=f"{param} Levels"
            )

            for _, row in latest_points.iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=6,
                    popup=f"<b>{param}:</b> {row[param]:.2f}<br><b>Date:</b> {row['date'].strftime('%Y-%m-%d')}",
                    color=colormap(row[param]),
                    fill=True, fill_color=colormap(row[param]), fill_opacity=0.75
                ).add_to(m)

            colormap.add_to(m)
            st_folium(m, key=param, use_container_width=True, height=500)

# ======================================================================================
# Main App Router
# ======================================================================================
if 'processed_data' not in st.session_state:
    render_landing_page()
else:
    render_dashboard()