import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Generate synthetic dataset
np.random.seed(42)

n = 5000

data = pd.DataFrame({
    "engine_size": np.random.uniform(1.0, 5.0, n),
    "mileage": np.random.uniform(8, 25, n),
    "distance_km": np.random.uniform(5, 500, n),
    "traffic_level": np.random.randint(1, 4, n),  # 1=Low, 2=Medium, 3=High
    "fuel_type": np.random.choice(["Petrol", "Diesel", "CNG"], n),
    "vehicle_type": np.random.choice(["SUV", "Sedan", "Hatchback"], n)
})

# Emission base factors
fuel_factor = {"Petrol": 2.31, "Diesel": 2.68, "CNG": 2.0}

# Generate realistic CO2
data["fuel_used"] = data["distance_km"] / data["mileage"]
data["co2"] = data.apply(
    lambda row: row["fuel_used"] *
    fuel_factor[row["fuel_type"]] *
    (1 + row["traffic_level"] * 0.1),
    axis=1
)

# One-hot encoding
data = pd.get_dummies(data, columns=["fuel_type", "vehicle_type"])

X = data.drop(["co2", "fuel_used"], axis=1)
y = data["co2"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

print("Model trained successfully")

# Save model
with open("emission_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model_columns.pkl", "wb") as f:
    pickle.dump(X.columns, f)

print("Model saved successfully")