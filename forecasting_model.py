import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import json

# Load the split data
df_train = pd.read_csv('train_data.csv')
df_test = pd.read_csv('test_data.csv')

# Convert 'ds' to datetime objects
df_train['ds'] = pd.to_datetime(df_train['ds'])
df_test['ds'] = pd.to_datetime(df_test['ds'])

# Define the regressors used in feature engineering
REGRESSORS = ['Fiscal_Balance', 'Current_Account_Balance', 'Inflation']

# Dictionary to store results
results = {
    'forecasts': {},
    'metrics': {},
    'components': {}
}

countries = df_train['Country'].unique()

for country in countries:
    print(f"\n--- Processing {country} ---")
    
    # Prepare data for the current country
    train_data = df_train[df_train['Country'] == country].copy()
    test_data = df_test[df_test['Country'] == country].copy()
    
    # Initialize and configure Prophet model
    # We will use the regressors as they were engineered
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False
    )
    
    # Add the regressors
    for regressor in REGRESSORS:
        model.add_regressor(regressor)
        
    # Fit the model
    model.fit(train_data[['ds', 'y'] + REGRESSORS])
    
    # --- 1. Model Evaluation on Test Set (2016-2020) ---
    
    # Create future dataframe for the test period
    future_test = test_data[['ds'] + REGRESSORS].copy()
    
    # Make prediction on the test set
    forecast_test = model.predict(future_test)
    
    # Merge actual values with forecast
    performance_df = pd.merge(
        test_data[['ds', 'y']], 
        forecast_test[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
        on='ds', 
        how='left'
    )
    
    # Calculate metrics
    y_true = performance_df['y'].values
    y_pred = performance_df['yhat'].values
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    # MAPE calculation (avoid division by zero)
    # Note: MAPE is very sensitive to values close to zero, which is the case for GDP growth.
    # We will use a robust version or just report the standard one with a warning.
    # For now, use the standard one.
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    results['metrics'][country] = {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }
    
    print(f"Evaluation Metrics on Test Set (2016-2020):")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    # --- 2. 5-Year Forecast (2021-2025) ---
    
    # Create future dataframe for the forecast period (2021-2025)
    future_years = pd.to_datetime([f'{y}-01-01' for y in range(2021, 2026)])
    future_forecast = pd.DataFrame({'ds': future_years})
    
    # Prophet requires regressors for the future period.
    # Since we don't have future values for the regressors, we will use the last known value (2020)
    # as a simple, naive forecast for the next 5 years. This is a common simplification in MVP.
    last_known_regressors = df_test[df_test['ds'].dt.year == 2020][REGRESSORS].iloc[0].to_dict()
    
    for regressor, value in last_known_regressors.items():
        future_forecast[regressor] = value
        
    # Make the 5-year forecast
    forecast_future = model.predict(future_forecast)
    
    # Combine historical, test, and future forecast data for visualization
    historical_data = train_data[['ds', 'y']].copy()
    historical_data['type'] = 'Historical (Train)'
    
    test_actual = test_data[['ds', 'y']].copy()
    test_actual['type'] = 'Historical (Test)'
    
    # Prepare forecast data for merging
    test_forecast_viz = performance_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={'yhat': 'y'}).copy()
    test_forecast_viz['type'] = 'Forecast (Test)'
    
    future_forecast_data = forecast_future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={'yhat': 'y'}).copy()
    future_forecast_data['type'] = 'Forecast (Future)'
    
    # Store the combined forecast data
    combined_forecast = pd.concat([historical_data, test_actual, test_forecast_viz, future_forecast_data], ignore_index=True)
    combined_forecast['Country'] = country
    
    # FIX: Convert 'ds' to string before saving to JSON
    combined_forecast['ds'] = combined_forecast['ds'].dt.strftime('%Y-%m-%d')
    
    results['forecasts'][country] = combined_forecast.to_dict(orient='records')
    
    # --- 3. Feature Importance & Interpretability (Prophet decomposition) ---
    
    # Prophet's decomposition is inherent in the model. We will extract the components.
    # We use the forecast_test dataframe as it contains the components for the test period.
    
    # The components are trend, yearly seasonality, and the regressors
    component_cols = ['ds', 'trend', 'yearly'] + REGRESSORS
    components_df = forecast_test[component_cols].copy()
    
    # Calculate the total regressor effect (sum of all regressor effects)
    # Prophet's output for regressors is directly their effect on yhat
    components_df['Regressors_Effect'] = components_df[REGRESSORS].sum(axis=1)
    
    # FIX: Convert 'ds' to string before saving to JSON
    components_df['ds'] = components_df['ds'].dt.strftime('%Y-%m-%d')
    
    results['components'][country] = components_df.to_dict(orient='records')
    
    print(f"Forecast for {country} stored.")

# Save all results to a JSON file
with open('forecasting_results.json', 'w') as f:
    json.dump(results, f, indent=4)
    
print("\nAll forecasting results (forecasts, metrics, components) saved to forecasting_results.json")
