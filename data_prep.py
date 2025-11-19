import pandas as pd

# Load the initial filtered data
df = pd.read_csv('initial_filtered_data.csv')

# Define the target KPI based on the instructions and the inspection of unique values
# The correct KPI is 'Real GDP growth (annual %)'
TARGET_KPI_NAME = 'Real GDP growth (annual %)'

# 1. Select the KPI: Real GDP growth (annual %).
gdp_growth_df = df[
    (df['Indicators Name'] == TARGET_KPI_NAME) | 
    (df['Indicators'] == TARGET_KPI_NAME)
].copy()

if gdp_growth_df.empty:
    print(f"Error: Could not find the KPI '{TARGET_KPI_NAME}' in the dataset even after correction.")
    exit()

# Drop unnecessary columns and keep only country, indicator name, and year columns
id_vars = ['Country and Regions Name', 'Indicators Name']
# Dynamically find year columns (assuming they are all 4-digit strings)
year_cols = [col for col in df.columns if col.isdigit() and len(col) == 4]
gdp_growth_df = gdp_growth_df[id_vars + year_cols]

# Rename the country column for simplicity
gdp_growth_df.rename(columns={'Country and Regions Name': 'Country'}, inplace=True)

# Pivot the data from wide to long format
df_long = pd.melt(
    gdp_growth_df, 
    id_vars=['Country', 'Indicators Name'], 
    value_vars=year_cols, 
    var_name='Year', 
    value_name='GDP_Growth'
)

# Convert 'Year' to integer and then to datetime for proper time-series handling
df_long['Year'] = df_long['Year'].astype(int)
df_long['ds'] = pd.to_datetime(df_long['Year'], format='%Y')

# Drop the 'Indicators Name' column as it's now redundant
df_long.drop(columns=['Indicators Name', 'Year'], inplace=True)

# Rename 'GDP_Growth' to 'y' for Prophet compatibility
df_long.rename(columns={'GDP_Growth': 'y'}, inplace=True)

# 2. Ensure the dataset is sorted chronologically per country.
df_long.sort_values(by=['Country', 'ds'], inplace=True)

# 3. Handle missing values appropriately: forward-fill or interpolate missing years.
# We will use linear interpolation for a smoother time series.
df_clean = df_long.groupby('Country').apply(
    lambda x: x.set_index('ds').interpolate(method='linear').reset_index()
).reset_index(drop=True)

# Check for any remaining NaNs
print("--- Missing values check after interpolation ---")
print(df_clean.isnull().sum())

# Final check of the cleaned data structure
print("\n--- Cleaned and Prepared Data Head ---")
print(df_clean.head())
print("\n--- Cleaned and Prepared Data Tail ---")
print(df_clean.tail())
print("\n--- Unique Years (ds) ---")
print(df_clean['ds'].dt.year.unique())

# Save the cleaned data for the next phase (EDA and Modeling)
df_clean.to_csv('gdp_growth_clean_data.csv', index=False)
print("\nCleaned and prepared GDP growth data saved to gdp_growth_clean_data.csv")
