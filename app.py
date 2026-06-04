import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("data/sales_data.csv")

# Features and target
X = df[["Month"]]
y = df["Sales"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Future prediction
future_months = pd.DataFrame({
    "Month": [13, 14, 15, 16, 17, 18]
})

predictions = model.predict(future_months)

# Display predictions
print("\nFuture Sales Forecast")

for month, sale in zip(future_months["Month"], predictions):
    print(f"Month {month}: {int(sale)}")

# Graph
plt.scatter(df["Month"], df["Sales"], label="Actual Sales")
plt.plot(df["Month"], model.predict(X), label="Trend Line")

plt.title("Sales Forecast")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()

plt.show()