import pandas as pd
import numpy as np
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor


# =====================================================
# 1. Load California Housing Dataset
# =====================================================

housing = fetch_california_housing(as_frame=True)

df = housing.frame

print("Dataset Preview:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)


# =====================================================
# 2. X and y Separate
# =====================================================

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

print("\nInput Features:")
print(X.columns)

print("\nTarget Column: MedHouseVal")


# =====================================================
# 3. Train-Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =====================================================
# 4. Create Pipeline
# =====================================================
# California dataset mainly numeric.
# SimpleImputer missing value handle করবে.
# RandomForestRegressor model train করবে.

model = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])


# =====================================================
# 5. Train Model
# =====================================================

model.fit(X_train, y_train)


# =====================================================
# 6. Prediction on Test Data
# =====================================================

y_pred = model.predict(X_test)


# =====================================================
# 7. Model Evaluation
# =====================================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)


# =====================================================
# 8. Save Trained Model with joblib
# =====================================================

joblib.dump(model, "california_house_price_model.pkl")

print("\nModel saved successfully as california_house_price_model.pkl")


# =====================================================
# 9. Load Saved Model
# =====================================================

loaded_model = joblib.load("california_house_price_model.pkl")

print("\nModel loaded successfully")


# =====================================================
# 10. New House Prediction
# =====================================================
# California dataset-এর original column গুলো:
# MedInc, HouseAge, AveRooms, AveBedrms, Population,
# AveOccup, Latitude, Longitude

new_house = pd.DataFrame({
    "MedInc": [8.3252],
    "HouseAge": [41.0],
    "AveRooms": [6.984],
    "AveBedrms": [1.023],
    "Population": [322.0],
    "AveOccup": [2.555],
    "Latitude": [37.88],
    "Longitude": [-122.23]
})

predicted_value = loaded_model.predict(new_house)

print("\nPredicted House Value:", predicted_value[0])

# California dataset target unit হলো 100,000 dollars
predicted_price_dollar = predicted_value[0] * 100000

print("Predicted House Price in Dollars:", predicted_price_dollar)