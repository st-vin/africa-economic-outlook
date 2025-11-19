import pandas as pd
import json
import numpy as np

# Load the forecasting results
with open('forecasting_results.json', 'r') as f:
    results = json.load(f)

# --- 1. Summarize Evaluation Metrics ---
print("--- Model Evaluation Summary (on Test Set 2016-2020) ---")
metrics_df = pd.DataFrame.from_dict(results['metrics'], orient='index')
metrics_df.index.name = 'Country'
print(metrics_df.to_markdown())

# --- 2. Prepare Feature Importance Data (Regressors' Average Absolute Effect) ---
# Prophet's "feature importance" is best represented by the magnitude of the regressor effects.
# We will calculate the average absolute effect of each regressor over the test period.

regressors = ['Fiscal_Balance', 'Current_Account_Balance', 'Inflation']
feature_importance_data = []

for country, components_list in results['components'].items():
    components_df = pd.DataFrame(components_list)
    
    # Calculate the average absolute effect for each regressor
    avg_abs_effects = components_df[regressors].abs().mean().to_dict()
    
    # Calculate the average absolute effect for the trend and seasonality for comparison
    avg_abs_effects['Trend'] = components_df['trend'].abs().mean()
    avg_abs_effects['Seasonality'] = components_df['yearly'].abs().mean()
    
    # Convert to a list of dictionaries for easy Altair visualization
    for feature, importance in avg_abs_effects.items():
        feature_importance_data.append({
            'Country': country,
            'Feature': feature,
            'Importance': importance
        })

feature_importance_df = pd.DataFrame(feature_importance_data)

# Save the feature importance data for the dashboard
feature_importance_df.to_csv('feature_importance_data.csv', index=False)
print("\nFeature importance data (average absolute effect) saved to feature_importance_data.csv")

# --- 3. Textual Insights on Model Performance and Interpretability ---
insights = """
--- Model Performance and Interpretability Insights ---

**Model Performance (Test Set 2016-2020):**
- **Kenya** shows the best performance with the lowest RMSE and MAE, and a reasonable MAPE (17.23%). This suggests the model is a good fit for Kenya's relatively stable time series.
- **South Africa** has a low RMSE/MAE, but a high MAPE (132.13%). This is a common issue when actual values (GDP growth) are close to zero, causing the percentage error to be inflated. The low absolute errors (MAE/RMSE) suggest the model is still useful.
- **Nigeria** has the worst performance by far (RMSE: 6.55, MAE: 6.38, MAPE: 445.45%). This confirms the earlier EDA finding that Nigeria's GDP growth is highly volatile and difficult to predict with a simple Prophet model, especially during the 2016-2020 period which included a major recession. The high error suggests a more complex model (like LSTM or a model incorporating more structural breaks) would be necessary for Nigeria.

**Feature Interpretability (Prophet Decomposition):**
- The feature importance data, based on the average absolute effect on the forecast, will be used to understand the drivers of the forecast.
- **Trend** and **Seasonality** (yearly) are the inherent components of the Prophet model.
- The **Regressors** (Fiscal Balance, Current Account Balance, Inflation) represent the external macroeconomic factors. Their relative importance compared to the inherent components will provide insight into whether external factors or internal time dynamics are the primary drivers of the forecast.
"""
print(insights)

# Save the insights to a file
with open('model_insights.txt', 'w') as f:
    f.write(insights)
print("Model insights saved to model_insights.txt")
