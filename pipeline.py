import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor


# =====================================================
# 1. Dataset
# =====================================================

df = pd.DataFrame({
    "Area": [1200, 1500, 1800, 1000, 2200, 1700, 1300, 2500, 1400, 2000],
    "Bedrooms": [2, 3, 3, 2, 4, 3, 2, 5, 3, 4],
    "Bathrooms": [1, 2, 2, 1, 3, 2, 1, 4, 2, 3],
    "HouseAge": [5, 10, 3, 8, 2, 6, 12, 1, 7, 4],
    "Location": ["Dhaka", "Gazipur", "Dhaka", "Narayanganj", "Dhaka",
                 "Gazipur", "Narayanganj", "Dhaka", "Gazipur", "Dhaka"],
    "HouseType": ["Apartment", "Apartment", "Duplex", "Apartment", "Villa",
                  "Duplex", "Apartment", "Villa", "Apartment", "Duplex"],
    "Price": [6500000, 7200000, 9500000, 4800000, 16000000,
              8800000, 5200000, 20000000, 7000000, 12000000]
})


# =====================================================
# 2. X and y
# =====================================================

X = df.drop("Price", axis=1)
y = df["Price"]


# =====================================================
# 3. Column types
# =====================================================

numeric_features = ["Area", "Bedrooms", "Bathrooms", "HouseAge"]
categorical_features = ["Location", "HouseType"]


# =====================================================
# 4. Numeric preprocessing pipeline
# =====================================================

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])


# =====================================================
# 5. Categorical preprocessing pipeline
# =====================================================

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])


# =====================================================
# 6. Combine preprocessing
# =====================================================

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])


# =====================================================
# 7. Final model pipeline
# =====================================================

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])


# =====================================================
# 8. Train test split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =====================================================
# 9. Model training
# =====================================================

model.fit(X_train, y_train)


# =====================================================
# 10. Prediction
# =====================================================

y_pred = model.predict(X_test)


# =====================================================
# 11. Evaluation
# =====================================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)


# =====================================================
# 12. New house prediction
# =====================================================

new_house = pd.DataFrame({
    "Area": [1600],
    "Bedrooms": [3],
    "Bathrooms": [2],
    "HouseAge": [4],
    "Location": ["Dhaka"],
    "HouseType": ["Apartment"]
})

predicted_price = model.predict(new_house)

print("Predicted House Price:", predicted_price[0])