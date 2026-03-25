# Sales Forecasting using ARIMA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA

# Load dataset
df = pd.read_csv("sales_data.csv")

# Convert Date column
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Sort data
df = df.sort_index()

# Plot original data
plt.figure(figsize=(10,5))
plt.plot(df['Sales'])
plt.title("Sales Over Time")
plt.show()

# Build ARIMA model
model = ARIMA(df['Sales'], order=(5,1,0))
model_fit = model.fit()

# Forecast next 30 days
forecast = model_fit.forecast(steps=30)

# Plot forecast
plt.figure(figsize=(10,5))
plt.plot(df['Sales'], label="Actual")
plt.plot(forecast, label="Forecast", color='red')
plt.legend()
plt.title("Sales Forecast")
plt.show()

# Save forecast
forecast.to_csv("forecast.csv")

print("Forecast completed successfully!")