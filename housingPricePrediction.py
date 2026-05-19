import pandas as pd
import numpy  as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


#1.load dataset

housing = fetch_california_housing(as_frame=True)
df = housing.frame

# print(df.head())
# print(df.info())
# print(df.isnull().sum())

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# print("Linear Regression Result")
# print("MAE:", mean_absolute_error(y_test, y_pred_lr))
# print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
# print("R2 Score:", r2_score(y_test, y_pred_lr))

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# print("\nRandom Forest Result")
# print("MAE:", mean_absolute_error(y_test, y_pred_rf))
# print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
# print("R2 Score:", r2_score(y_test, y_pred_rf))


new_house = [[8.3252, 41.0, 6.984, 1.023, 322.0, 2.555, 37.88, -122.23]]
predicted_price = rf_model.predict(new_house)

print("\nPredicted House Value:", predicted_price[0])
