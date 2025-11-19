import streamlit as st
import pandas as pd
import altair as alt
import json
import numpy as np

# --- Configuration ---
st.set_page_config(
    page_title="African GDP Growth Forecasting MVP",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Data Loading ---
@st.cache_data
def load_data():
    """Loads all necessary data for the dashboard."""
    try:
        # Load forecasting results
        with open('forecasting_results.json', 'r') as f:
            results = json.load(f)
        
        # Convert forecasts to a single DataFrame
        all_forecasts = []
        for country, data in results['forecasts'].items():
            df = pd.DataFrame(data)
            df['ds'] = pd.to_datetime(df['ds'])
            all_forecasts.append(df)
        df_forecast = pd.concat(all_forecasts, ignore_index=True)
        
        # Load feature importance
        df_importance = pd.read_csv('feature_importance_data.csv')
        
        # Load model metrics
        df_metrics = pd.DataFrame.from_dict(results['metrics'], orient='index')
        df_metrics.index.name = 'Country'
        df_metrics = df_metrics.reset_index()
        
        # Load insights
        with open('eda_insights.txt', 'r') as f:
            eda_insights = f.read()
        with open('model_insights.txt', 'r') as f:
            model_insights = f.read()
            
        return df_forecast, df_importance, df_metrics, eda_insights, model_insights
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None

df_forecast, df_importance, df_metrics, eda_insights, model_insights = load_data()

if df_forecast is None:
    st.stop()

# --- Sidebar ---
st.sidebar.title("Dashboard Controls")
countries = df_forecast['Country'].unique().tolist()
selected_country = st.sidebar.selectbox("Select Country", countries)

# --- Main Content ---
st.title(f"GDP Growth Forecasting MVP: {selected_country}")
st.markdown("A time-series forecasting application for the GDP growth of key African economies.")

# Filter data for the selected country
df_country = df_forecast[df_forecast['Country'] == selected_country].copy()
df_country_metrics = df_metrics[df_metrics['Country'] == selected_country].iloc[0]
df_country_importance = df_importance[df_importance['Country'] == selected_country].copy()

# --- 1. Historical Trends and Forecast ---
st.header("1. Historical Trends and 5-Year Forecast (2021-2025)")

# Separate historical and forecast data for visualization
df_historical = df_country[df_country['type'].isin(['Historical (Train)', 'Historical (Test)'])].copy()
df_forecast_viz = df_country[df_country['type'].isin(['Forecast (Test)', 'Forecast (Future)'])].copy()

# Create the base chart
base = alt.Chart(df_country).encode(
    x=alt.X('ds:T', title='Year')
)

# Historical line (Actuals)
historical_line = base.mark_line(point=True).encode(
    y=alt.Y('y:Q', title='GDP Growth (Annual %)'),
    color=alt.value('darkblue'),
    tooltip=[alt.Tooltip('ds:T', title='Year', format='%Y'), alt.Tooltip('y:Q', title='Actual Growth', format='.2f')]
).transform_filter(
    alt.FieldOneOfPredicate(field='type', oneOf=['Historical (Train)', 'Historical (Test)'])
)

# Forecast line (Predicted)
forecast_line = alt.Chart(df_forecast_viz).mark_line(point=True, strokeDash=[5, 5]).encode(
    x=alt.X('ds:T'),
    y=alt.Y('y:Q'),
    color=alt.value('red'),
    tooltip=[alt.Tooltip('ds:T', title='Year', format='%Y'), alt.Tooltip('y:Q', title='Forecasted Growth', format='.2f')]
)

# Confidence Interval (Area)
confidence_interval = alt.Chart(df_forecast_viz).mark_area(opacity=0.3, color='red').encode(
    x=alt.X('ds:T'),
    y='yhat_lower:Q',
    y2='yhat_upper:Q',
    tooltip=[alt.Tooltip('yhat_lower:Q', title='Lower Bound', format='.2f'), alt.Tooltip('yhat_upper:Q', title='Upper Bound', format='.2f')]
)

# Combine charts
chart = (confidence_interval + historical_line + forecast_line).properties(
    title=f'GDP Growth: Historical vs. Forecast for {selected_country}'
).interactive()

st.altair_chart(chart, use_container_width=True)

# --- 2. Model Performance and Feature Importance ---
col1, col2 = st.columns(2)

with col1:
    st.header("2. Model Performance (Test Set 2016-2020)")
    st.metric(label="Root Mean Squared Error (RMSE)", value=f"{df_country_metrics['RMSE']:.4f}")
    st.metric(label="Mean Absolute Error (MAE)", value=f"{df_country_metrics['MAE']:.4f}")
    st.metric(label="Mean Absolute Percentage Error (MAPE)", value=f"{df_country_metrics['MAPE']:.2f}%")
    
    st.markdown(f"""
    **Interpretation:**
    - The model's average absolute error (MAE) is **{df_country_metrics['MAE']:.2f}** percentage points.
    - The high MAPE for countries like Nigeria and South Africa is likely due to actual GDP growth values being close to zero in the test period, which inflates the percentage error.
    """)

with col2:
    st.header("3. Feature Importance (Average Absolute Effect)")
    
    # Sort by importance and rename features for display
    df_country_importance['Feature'] = df_country_importance['Feature'].replace({
        'Fiscal_Balance': 'Fiscal Balance',
        'Current_Account_Balance': 'Current Account Balance',
        'Inflation': 'Inflation',
        'Trend': 'Prophet Trend',
        'Seasonality': 'Prophet Seasonality'
    })
    
    # Bar chart for feature importance
    importance_chart = alt.Chart(df_country_importance).mark_bar().encode(
        x=alt.X('Importance:Q', title='Average Absolute Effect on Forecast'),
        y=alt.Y('Feature:N', sort='-x', title='Feature'),
        tooltip=['Feature', alt.Tooltip('Importance:Q', format='.4f')]
    ).properties(
        title=f'Drivers of GDP Growth Forecast for {selected_country}'
    )
    
    st.altair_chart(importance_chart, use_container_width=True)

# --- 4. Country Comparison and Insights ---
st.header("4. Country Comparison and Key Insights")

# Comparison Table (Metrics)
st.subheader("Model Performance Comparison")
st.dataframe(df_metrics.set_index('Country').style.format({'RMSE': '{:.4f}', 'MAE': '{:.4f}', 'MAPE': '{:.2f}%'}), use_container_width=True)

# Comparison Chart (Historical) - Re-using the historical line chart logic for all countries
st.subheader("Historical GDP Growth Comparison (1981-2020)")
base_comp = alt.Chart(df_historical).encode(
    x=alt.X('ds:T', title='Year'),
    y=alt.Y('y:Q', title='GDP Growth (Annual %)'),
    color=alt.Color('Country:N'),
    tooltip=[alt.Tooltip('ds:T', title='Year', format='%Y'), alt.Tooltip('y:Q', title='Growth', format='.2f'), 'Country']
).properties(
    title='Historical GDP Growth: Kenya vs. Nigeria vs. South Africa'
).interactive()

st.altair_chart(base_comp.mark_line(point=True), use_container_width=True)

# Textual Insights
st.subheader("Textual Insights")
with st.expander("Expand for Exploratory Data Analysis (EDA) Insights"):
    st.markdown(eda_insights)
    
with st.expander("Expand for Model Performance and Forecast Insights"):
    st.markdown(model_insights)
    
# --- Optional: Allow download of forecast data as CSV ---
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(df_country)

st.sidebar.download_button(
    label="Download Forecast Data as CSV",
    data=csv,
    file_name=f'{selected_country}_gdp_forecast.csv',
    mime='text/csv',
)

st.sidebar.markdown("---")
st.sidebar.info("Dashboard built using Streamlit and Altair for interactive visualization.")
