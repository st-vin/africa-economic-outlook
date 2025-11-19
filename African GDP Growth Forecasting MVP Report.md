# African GDP Growth Forecasting MVP Report

## **Objective**

The goal of this project was to build a **Minimum Viable Product (MVP)** for time-series forecasting of **GDP growth (annual % change)** for three key African economies: **Kenya, South Africa, and Nigeria**. The project followed an end-to-end data science workflow, including data preparation, exploratory data analysis (EDA), feature engineering, model training, evaluation, and the creation of an interactive dashboard for visualization and interpretability.

## **1. Data Loading and Preparation**

The analysis was performed on the provided `african-economic-outlook.csv` dataset.

| Step | Description | Outcome |
| :--- | :--- | :--- |
| **Filtering** | Data was filtered for the target countries: Kenya, South Africa, and Nigeria. | 87 rows of multi-indicator data retained. |
| **KPI Selection** | The target variable was identified as **'Real GDP growth (annual %)'**. | Data subsetted to only include the target KPI. |
| **Data Transformation** | Data was pivoted from wide (years as columns) to long format (single 'ds' and 'y' columns) for time-series modeling. | Ready for Prophet modeling. |
| **Missing Value Handling** | Linear interpolation was applied to fill minor gaps in the time series. The initial 1980 data point for each country was dropped due to missing values. | Cleaned dataset from 1981 to 2020. |

## **2. Exploratory Data Analysis (EDA) Summary**

The EDA revealed significant differences in the economic growth patterns of the three countries:

| Country | Mean GDP Growth (%) | Standard Deviation | Key Trend |
| :--- | :--- | :--- | :--- |
| **Nigeria** | 4.23 | 6.27 | Highest average growth but also the highest volatility, with sharp peaks and troughs (e.g., oil price shocks). |
| **Kenya** | 4.06 | 2.31 | Moderate and relatively stable growth, with a steady upward trend in the 2000s. |
| **South Africa** | 2.15 | 2.12 | Lowest average growth and volatility, suggesting a more mature and slower-growing economy with a prolonged period of stagnation post-2008. |

## **3. Feature Engineering and Train-Test Split**

To enhance the model, three macroeconomic indicators were included as **exogenous regressors** in the Prophet model:

1. **Fiscal Balance (% of GDP)**
2. **Current Account Balance (As % of GDP)**
3. **Inflation (consumer prices, annual %)**

The time series was split chronologically to maintain integrity:

* **Training Set:** 1981–2015 (102 data points)
* **Testing Set:** 2016–2020 (15 data points)

## **4. Model Training and Evaluation**

The **Prophet** model was trained separately for each country, incorporating the three exogenous regressors. A 5-year forecast (2021–2025) was generated using the last known values (2020) for the regressors as a simple assumption for the future.

### **Model Performance (Test Set: 2016–2020)**

| Country | RMSE | MAE | MAPE | Insight |
| :--- | :--- | :--- | :--- | :--- |
| **Kenya** | 1.09 | 0.98 | 17.23% | Best performance, indicating a good fit for its stable growth pattern. |
| **South Africa** | 1.19 | 1.01 | 132.13% | Low absolute errors (MAE/RMSE) but high MAPE, which is common when actual growth is near zero. Model is still reasonably accurate in absolute terms. |
| **Nigeria** | 6.55 | 6.38 | 445.45% | Poor performance due to the country's extreme volatility and the major recession in the test period. A more complex model (e.g., one accounting for structural breaks) is recommended. |

### **Feature Interpretability**

The dashboard includes a feature importance chart based on the **average absolute effect** of each component (Trend, Seasonality, Regressors) on the forecast. This allows stakeholders to understand the primary drivers of the predicted GDP growth for each country.

## **5. Interactive Dashboard**

A fully functional interactive dashboard was created using **Streamlit** and **Altair** for visualization.

### **Dashboard Features:**

* **Country Selection:** Dropdown to select Kenya, South Africa, or Nigeria.
* **Historical & Forecast Chart:** Interactive line chart showing historical GDP growth (1981-2020) and the 5-year forecast (2021-2025) with confidence intervals.
* **Model Metrics:** Display of RMSE, MAE, and MAPE for the selected country.
* **Feature Importance:** Bar chart showing the relative importance of the Prophet components (Trend, Seasonality) and the macroeconomic regressors.
* **Insights:** Textual summaries of the EDA and model performance.
* **Data Download:** Option to download the forecast data as a CSV file.

## **6. Deliverables**

The following files are provided as the final deliverables:

1. **`forecasting_report.md`**: This final report (current file).
2. **`app.py`**: The complete Streamlit dashboard code.
3. **`forecasting_results.json`**: Raw output of the model training, including forecasts, metrics, and component data for all countries.
4. **`historical_gdp_growth.png`**: Static visualization of the historical GDP growth from the EDA phase.
5. **`gdp_growth_final_clean_data.csv`**: The final cleaned and prepared dataset used for modeling.
6. **Interactive Dashboard Link**: The live link to the Streamlit application.
