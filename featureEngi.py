import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# ==========================================
# 1. Load Dataset
# ==========================================

housing = fetch_california_housing(as_frame=True)
df = housing.frame

print("Original Data:")
print(df.head())
print(df.shape)


# ==========================================
# 2. X and y separate
# ==========================================

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]


# ==========================================
# 3. Feature Engineering Function
# ==========================================
# এই function-এর ভিতরে আমরা নতুন column বানাচ্ছি

def add_features(data):
    data = data.copy()

    # New Feature 1:
    # প্রতি household-এ average কত room আছে
    data["rooms_per_household"] = data["AveRooms"] / data["AveOccup"]

    # New Feature 2:
    # total room-এর তুলনায় bedroom ratio কেমন
    data["bedrooms_per_room"] = data["AveBedrms"] / data["AveRooms"]

    # New Feature 3:
    # প্রতি household-এ population density কেমন
    data["population_per_household"] = data["Population"] / data["AveOccup"]

    # New Feature 4:
    # income এবং room size combine করে নতুন relation
    data["income_per_room"] = data["MedInc"] / data["AveRooms"]

    return data


# ==========================================
# 4. Apply Feature Engineering
# ==========================================
# এখানে X-এর উপর নতুন feature add করা হলো

X_engineered = add_features(X)

print("\nData After Feature Engineering:")
print(X_engineered.head())
print(X_engineered.shape)


# ==========================================
# 5. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_engineered,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 6. Linear Regression Model
# ==========================================

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)


# ==========================================
# 7. Linear Regression Evaluation
# ==========================================

lr_mae = mean_absolute_error(y_test, y_pred_lr)
lr_mse = mean_squared_error(y_test, y_pred_lr)
lr_rmse = np.sqrt(lr_mse)
lr_r2 = r2_score(y_test, y_pred_lr)

print("\nLinear Regression Result With Feature Engineering")
print("MAE:", lr_mae)
print("RMSE:", lr_rmse)
print("R2 Score:", lr_r2)


# ==========================================
# 8. Random Forest Model
# ==========================================

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)


# ==========================================
# 9. Random Forest Evaluation
# ==========================================

rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Result With Feature Engineering")
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("R2 Score:", rf_r2)


# ==========================================
# 10. New House Prediction
# ==========================================
# New data দিতে হবে original column format অনুযায়ী

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

# New data-তেও একই feature engineering apply করতে হবে
new_house_engineered = add_features(new_house)

predicted_price = rf_model.predict(new_house_engineered)

print("\nPredicted House Value:", predicted_price[0])