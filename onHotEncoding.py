import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor


# ==================================================
# 1. Example Dataset
# ==================================================
# আপনার real dataset থাকলে এই অংশের বদলে pd.read_csv() ব্যবহার করবেন

df = pd.DataFrame({
    "Area": [1200, 1500, 1800, 1000, 2200, 1700, 1300, 2500],
    "Bedrooms": [2, 3, 3, 2, 4, 3, 2, 5],
    "Bathrooms": [1, 2, 2, 1, 3, 2, 1, 4],
    "HouseAge": [5, 10, 3, 8, 2, 6, 12, 1],
    "Location": ["Dhaka", "Gazipur", "Dhaka", "Narayanganj", "Dhaka", "Gazipur", "Narayanganj", "Dhaka"],
    "HouseType": ["Apartment", "Apartment", "Duplex", "Apartment", "Villa", "Duplex", "Apartment", "Villa"],
    "Furnishing": ["Furnished", "Semi-Furnished", "Furnished", "Unfurnished", "Furnished", "Semi-Furnished", "Unfurnished", "Furnished"],
    "Price": [6500000, 7200000, 9500000, 4800000, 16000000, 8800000, 5200000, 20000000]
})

# Real CSV হলে এমন হবে:
# df = pd.read_csv("house_data.csv")


# ==================================================
# 2. X and y separate
# ==================================================

X = df.drop("Price", axis=1)
y = df["Price"]


# ==================================================
# 3. Numeric and Categorical Column আলাদা করা
# ==================================================

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object", "category"]).columns

print("Numeric Columns:", numeric_features)
print("Categorical Columns:", categorical_features)


# ==================================================
# 4. Numeric data processing
# ==================================================
# Missing numeric value থাকলে median দিয়ে fill করবে

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])


# ==================================================
# 5. Categorical data processing with One-Hot Encoding
# ==================================================
# Missing categorical value থাকলে most frequent value দিয়ে fill করবে
# তারপর OneHotEncoder apply করবে

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])


# ==================================================
# 6. ColumnTransformer
# ==================================================
# Numeric column-এ numeric_transformer apply হবে
# Categorical column-এ categorical_transformer apply হবে

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ==================================================
# 7. Full Pipeline
# ==================================================
# Pipeline-এর ভিতরে:
# First: preprocessing
# Second: model training

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])


# ==================================================
# 8. Train-Test Split
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==================================================
# 9. Model Training
# ==================================================

model.fit(X_train, y_train)


# ==================================================
# 10. Prediction on Test Data
# ==================================================

y_pred = model.predict(X_test)


# ==================================================
# 11. Model Evaluation
# ==================================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)


# ==================================================
# 12. New House Price Prediction
# ==================================================
# Important:
# New data-তে original column দিবেন।
# One-hot encoding manually করতে হবে না।
# Pipeline নিজে automatically করবে।

new_house = pd.DataFrame({
    "Area": [1600],
    "Bedrooms": [3],
    "Bathrooms": [2],
    "HouseAge": [4],
    "Location": ["Dhaka"],
    "HouseType": ["Apartment"],
    "Furnishing": ["Furnished"]
})

predicted_price = model.predict(new_house)

print("\nPredicted House Price:", predicted_price[0])