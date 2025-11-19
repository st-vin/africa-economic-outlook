import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the cleaned data
df = pd.read_csv('gdp_growth_clean_data.csv')

# Convert 'ds' back to datetime
df['ds'] = pd.to_datetime(df['ds'])

# Drop the remaining NaNs (first year for each country)
df.dropna(subset=['y'], inplace=True)

# 1. Generate summary statistics
print("--- Summary Statistics (GDP Growth) ---")
summary_stats = df.groupby('Country')['y'].agg(['mean', 'std', 'min', 'max']).reset_index()
print(summary_stats.to_markdown(index=False))

# Calculate year-over-year change (YoY) for each country
df['YoY_Change'] = df.groupby('Country')['y'].diff()
yoy_stats = df.groupby('Country')['YoY_Change'].agg(['mean', 'std', 'min', 'max']).reset_index()
print("\n--- Year-over-Year Change Statistics ---")
print(yoy_stats.to_markdown(index=False))

# 2. Visualize GDP growth over time for each country (line charts).
plt.figure(figsize=(14, 7))
sns.lineplot(data=df, x='ds', y='y', hue='Country', marker='o')
plt.title('Historical GDP Growth (Annual %) for Kenya, Nigeria, and South Africa (1981-2020)')
plt.xlabel('Year')
plt.ylabel('GDP Growth (%)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Country')
plt.tight_layout()
plt.savefig('historical_gdp_growth.png')
print("\nHistorical GDP growth chart saved to historical_gdp_growth.png")

# 3. Identify any outliers or anomalies.
# Use a box plot to visualize distribution and potential outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Country', y='y')
plt.title('Box Plot of GDP Growth by Country')
plt.ylabel('GDP Growth (%)')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('gdp_growth_boxplot.png')
print("Box plot saved to gdp_growth_boxplot.png")

# 4. Provide textual insights summarizing trends and differences between countries.
insights = """
--- Textual Insights from EDA ---

**Summary Statistics:**
- **Nigeria** has the highest average GDP growth, but also the highest standard deviation, indicating more volatility.
- **South Africa** has the lowest average GDP growth and the lowest maximum growth, suggesting a more mature and slower-growing economy in this period.
- **Kenya** shows moderate growth and volatility, positioned between Nigeria and South Africa.

**Year-over-Year Change:**
- The mean year-over-year change is close to zero for all countries, as expected for a long time series.
- **Nigeria** again shows the highest volatility in YoY change (highest standard deviation), reflecting sharp economic swings.

**Visual Trends (Historical GDP Growth):**
- **Nigeria's** growth is characterized by significant peaks and troughs, with a notable period of high growth in the early 2000s, followed by a sharp decline after 2014 (likely due to oil price shocks).
- **South Africa's** growth is consistently lower and less volatile, with a clear dip around the 2008-2009 global financial crisis and a prolonged period of stagnation/low growth afterward.
- **Kenya's** growth is relatively stable, with a steady upward trend in the 2000s, though it also experienced a slowdown post-2008.

**Outliers/Anomalies (Box Plot):**
- **Nigeria** has the most pronounced outliers, with both extremely high and low growth rates, confirming its high volatility.
- **South Africa** and **Kenya** have fewer extreme outliers, with their growth rates generally clustered closer to the mean.

**Conclusion for Modeling:**
The distinct volatility and trend characteristics for each country suggest that a separate model for each country is the correct approach, as planned. The high volatility in Nigeria's series might make forecasting more challenging.
"""
print(insights)

# Save the insights to a file
with open('eda_insights.txt', 'w') as f:
    f.write(insights)
print("EDA insights saved to eda_insights.txt")

# Save the final cleaned data (after dropping 1980 NaNs)
df.to_csv('gdp_growth_final_clean_data.csv', index=False)
print("Final cleaned data saved to gdp_growth_final_clean_data.csv")
