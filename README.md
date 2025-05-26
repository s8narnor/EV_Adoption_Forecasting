# ⚡ Forecasting EV Adoption in Washington State

EV_Adoption_Forecasting is a machine learning-powered forecasting tool built to predict electric vehicle (EV) adoption trends across counties in Washington State. The app provides a 3-year forecast of cumulative EV registrations using historical data and trend-based features.

## 🚀 Project Goals

- Predict EV growth across different counties using regression models
- Provide interactive visualizations via Streamlit
- Support planning for infrastructure like EV charging stations

## 📊 Features Used

- EV counts (lags, rolling averages, growth slope)
- County encoding
- Time features (months since start)
- Historical cumulative EV totals

## 🧠 Model Info

- Model Type: Random Forest (replace as needed)
- Evaluation Metric: R² score
- Trained on 2017–2024 EV population data by county

## 📈 Forecast Output

- Monthly EV forecasts for the next 3 years
- Visualization of cumulative EV growth
- Summary of last 12 months' growth %

## 🖥️ Streamlit App
To run on cloud - [ev-adoption-forecast](https://ev-adoption-forecast-wa.streamlit.app/)

To run the app locally:

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
