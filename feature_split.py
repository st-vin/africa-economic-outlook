import pandas as pd

# Load the final cleaned GDP growth data
df_gdp = pd.read_csv('gdp_growth_final_clean_data.csv')
df_gdp['ds'] = pd.to_datetime(df_gdp['ds'])

# Load the initial filtered data to check for other indicators (potential regressors)
df_all_indicators = pd.read_csv('initial_filtered_data.csv')

# Identify potential regressors: other indicators that are not GDP growth
gdp_kpi_name = 'Real GDP growth (annual %)'
regressor_indicators = df_all_indicators[
    df_all_indicators['Indicators Name'] != gdp_kpi_name
]['Indicators Name'].unique()

print("--- Potential Regressor Indicators ---")
print(regressor_indicators)

# For simplicity and to follow the core instruction of using Prophet, we will select a few key indicators
# that are likely to influence GDP growth, if they are present and complete.
# Let's select 'Inflation, consumer prices (annual %)' and 'Current account balance (As % of GDP)'
# as common macroeconomic features.

selected_regressors = [
    'Inflation, consumer prices (annual %)',
    'Current account balance (As % of GDP)',
    'Central government, Fiscal Balance (% of GDP)'
]

# Prepare the regressor data
df_regressors = df_all_indicators[
    df_all_indicators['Indicators Name'].isin(selected_regressors)
].copy()

# Pivot the regressor data to long format
id_vars = ['Country and Regions Name', 'Indicators Name']
year_cols = [col for col in df_all_indicators.columns if col.isdigit() and len(col) == 4]
df_regressors = df_regressors[id_vars + year_cols]
df_regressors.rename(columns={'Country and Regions Name': 'Country'}, inplace=True)

df_regressors_long = pd.melt(
    df_regressors, 
    id_vars=['Country', 'Indicators Name'], 
    value_vars=year_cols, 
    var_name='Year', 
    value_name='Value'
)

df_regressors_long['ds'] = pd.to_datetime(df_regressors_long['Year'].astype(int), format='%Y')
df_regressors_long.drop(columns=['Year'], inplace=True)

# Pivot the regressor data so each indicator is a column
df_regressors_pivot = df_regressors_long.pivot_table(
    index=['Country', 'ds'], 
    columns='Indicators Name', 
    values='Value'
).reset_index()

# Clean up column names
df_regressors_pivot.columns.name = None
df_regressors_pivot.rename(columns={
    'Inflation, consumer prices (annual %)': 'Inflation',
    'Current account balance (As % of GDP)': 'Current_Account_Balance',
    'Central government, Fiscal Balance (% of GDP)': 'Fiscal_Balance'
}, inplace=True)

# Merge GDP data with regressors
df_merged = pd.merge(df_gdp, df_regressors_pivot, on=['Country', 'ds'], how='left')

# Handle missing values in regressors: linear interpolation within each country
df_final = df_merged.groupby('Country').apply(
    lambda x: x.set_index('ds').interpolate(method='linear').reset_index()
).reset_index(drop=True)

# Drop any remaining NaNs (e.g., at the start of the series)
df_final.dropna(inplace=True)

print("\n--- Final DataFrame with Regressors Head (Multivariate Model Ready) ---")
print(df_final.head())
print("\n--- Final DataFrame with Regressors Info ---")
print(df_final.info())

# Save the final multivariate dataset
df_final.to_csv('gdp_growth_multivariate_data.csv', index=False)
print("\nMultivariate data saved to gdp_growth_multivariate_data.csv")

# --- Train-Test Split ---
# Training: 1980–2015
# Testing: 2016–2020

TRAIN_END_YEAR = 2015
TEST_START_YEAR = 2016
TEST_END_YEAR = 2020

df_train = df_final[df_final['ds'].dt.year <= TRAIN_END_YEAR].copy()
df_test = df_final[
    (df_final['ds'].dt.year >= TEST_START_YEAR) & 
    (df_final['ds'].dt.year <= TEST_END_YEAR)
].copy()

print(f"\n--- Train-Test Split Summary ---")
print(f"Training set size: {len(df_train)} rows (Years up to {TRAIN_END_YEAR})")
print(f"Testing set size: {len(df_test)} rows (Years {TEST_START_YEAR} to {TEST_END_YEAR})")

# Save the split datasets
df_train.to_csv('train_data.csv', index=False)
df_test.to_csv('test_data.csv', index=False)
print("Train and test data saved to train_data.csv and test_data.csv")
