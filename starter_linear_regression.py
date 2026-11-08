# ML 402331 - Phase 1 Starter Code (Smart Room Occupancy)
# Student: Abdallah Ayyad (2310216)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the sensor dataset (550 rows)
df = pd.read_csv("room_occupancy_data.csv")
print(df.head())

# Step 2: Pick Input Features (X) and Target Label (y)
X = df[["Temperature", "Humidity", "Light", "CO2"]]
y = df["People_Count"]

# Step 3: Split into Training Data (first 440 rows) and Testing Data (last 110 rows)
X_train = X[:440]
X_test = X[440:]
y_train = y[:440]
y_test = y[440:]

# Step 4: Create and Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Test the Model and Print the Results
predictions = model.predict(X_test)
print("Intercept (w0):", model.intercept_)
print("Weights (w1, w2, w3, w4):", model.coef_)
print("Mean Squared Error (MSE):", mean_squared_error(y_test, predictions))
print("R2 Accuracy Score:", r2_score(y_test, predictions))

# Step 6: Plot Real vs. Predicted Values
plt.scatter(y_test, predictions)
plt.xlabel("Real People Count (y)")
plt.ylabel("Predicted People Count (h(x))")
plt.title("Linear Regression: Real vs. Predicted Room Occupancy")
plt.show()
