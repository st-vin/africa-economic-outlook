import pandas as pd
import os

# Define the path to the uploaded file
file_path = 'african-economic-outlook.csv'

# Check if the file exists
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
else:
    # 1. Load the provided African Economic Outlook CSV dataset.
    df = pd.read_csv(file_path)

    # Display initial information
    print("--- Initial DataFrame Info ---")
    print(df.info())
    print("\n--- First 5 Rows ---")
    print(df.head())

    # Define the target countries
    target_countries = ['Kenya', 'South Africa', 'Nigeria']

    # 2. Filter the data for the countries: Kenya, South Africa, Nigeria.
    # Assuming the country column is named 'Country' or similar. We'll check the columns.
    
    # Identify the country column (assuming it's the one with the country names)
    country_col = None
    for col in df.columns:
        # Simple heuristic: check if any of the target countries are in the first few non-null values
        if df[col].astype(str).str.contains('|'.join(target_countries)).any():
            country_col = col
            break
    
    if country_col:
        print(f"\n--- Identified Country Column: {country_col} ---")
        df_filtered = df[df[country_col].isin(target_countries)].copy()
        
        print("\n--- Filtered DataFrame Shape ---")
        print(df_filtered.shape)
        print("\n--- Filtered DataFrame Countries ---")
        print(df_filtered[country_col].unique())
        print("\n--- Filtered DataFrame Columns ---")
        print(df_filtered.columns.tolist())
        
        # Save the filtered data to an intermediate file for the next step
        df_filtered.to_csv('initial_filtered_data.csv', index=False)
        print("\nInitial filtered data saved to initial_filtered_data.csv")
    else:
        print("\nCould not automatically identify the country column. Please inspect the head of the DataFrame.")
        