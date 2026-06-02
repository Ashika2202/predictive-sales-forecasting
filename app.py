import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# -------------------------
# Load Dataset
# -------------------------

df = pd.read_csv("data/sales_data.csv")

print("\nOriginal Dataset")
print(df)

# -------------------------
# Prepare Data
# -------------------------

X = df[['Month']]
y = df['Sales']

# -------------------------
# Train Model
# -------------------------

model = LinearRegression()

model.fit(X, y)

# -------------------------
# Predict Existing Data
# -------------------------

predicted_sales = model.predict(X)

# -------------------------
# Evaluation
# -------------------------

mae = mean_absolute_error(y, predicted_sales)

mse = mean_squared_error(y, predicted_sales)

r2 = r2_score(y, predicted_sales)

print("\nModel Performance")

print("MAE :", round(mae,2))
print("MSE :", round(mse,2))
print("R2 Score :", round(r2,4))

# -------------------------
# Future Forecast
# -------------------------

future_months = pd.DataFrame({
    'Month':[13,14,15,16,17,18]
})

future_predictions = model.predict(future_months)

forecast_df = pd.DataFrame({
    'Future Month':future_months['Month'],
    'Predicted Sales':future_predictions.astype(int)
})

print("\nFuture Sales Forecast")
print(forecast_df)

# -------------------------
# Save Forecast
# -------------------------

forecast_df.to_csv(
    "output/forecast_results.csv",
    index=False
)

# -------------------------
# Visualization
# -------------------------

plt.figure(figsize=(10,6))

plt.scatter(
    X,
    y,
    label="Actual Sales"
)

plt.plot(
    X,
    predicted_sales,
    label="Regression Line"
)

plt.scatter(
    future_months,
    future_predictions,
    label="Forecast"
)

plt.title("Predictive Sales Forecast")

plt.xlabel("Month")

plt.ylabel("Sales")

plt.legend()

plt.grid(True)

plt.savefig("output/forecast_plot.png")

plt.show()

print("\nForecast saved successfully.")