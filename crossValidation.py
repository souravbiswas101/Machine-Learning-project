import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, KFold
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
    "Area": [1200, 1500, 1800, 1000, 2200, 1700, 1300, 2500, 1400, 2000,
             1100, 1900, 2100, 1600, 1750, 2300, 1250, 1550, 2400, 1350],
    
    "Bedrooms": [2, 3, 3, 2, 4, 3, 2, 5, 3, 4,
                 2, 3, 4, 3, 3, 4, 2, 3, 5, 2],
    
    "Bathrooms": [1, 2, 2, 1, 3, 2, 1, 4, 2, 3,
                  1, 2, 3, 2, 2, 3, 1, 2, 4, 1],
    
    "HouseAge": [5, 10, 3, 8, 2, 6, 12, 1, 7, 4,
                 15, 5, 3, 6, 8, 2, 11, 7, 1, 9],
    
    "Location": ["Dhaka", "Gazipur", "Dhaka", "Narayanganj", "Dhaka",
                 "Gazipur", "Narayanganj", "Dhaka", "Gazipur", "Dhaka",
                 "Narayanganj", "Dhaka", "Dhaka", "Gazipur", "Narayanganj",
                 "Dhaka", "Gazipur", "Gazipur", "Dhaka", "Narayanganj"],
    
    "HouseType": ["Apartment", "Apartment", "Duplex", "Apartment", "Villa",
                  "Duplex", "Apartment", "Villa", "Apartment", "Duplex",
                  "Apartment", "Duplex", "Villa", "Apartment", "Duplex",
                  "Villa", "Apartment", "Duplex", "Villa", "Apartment"],
    
    "Price": [6500000, 7200000, 9500000, 4800000, 16000000,
              8800000, 5200000, 20000000, 7000000, 12000000,
              4300000, 11000000, 15000000, 7600000, 8500000,
              17500000, 5800000, 8000000, 19000000, 5500000]
})


# =====================================================
# 2. X and y separate
# =====================================================

X = df.drop("Price", axis=1)
y = df["Price"]


# =====================================================
# 3. Numeric and categorical columns
# =====================================================

numeric_features = ["Area", "Bedrooms", "Bathrooms", "HouseAge"]
categorical_features = ["Location", "HouseType"]


# =====================================================
# 4. Numeric pipeline
# =====================================================

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])


# =====================================================
# 5. Categorical pipeline
# =====================================================

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])


# =====================================================
# 6. Preprocessor
# =====================================================

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])


# =====================================================
# 7. Final pipeline: Preprocessing + Model
# =====================================================

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])


# =====================================================
# 8. Cross-validation
# =====================================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=kf,
    scoring="r2"
)

print("Cross-validation R2 scores:", cv_scores)
print("Average CV R2 Score:", cv_scores.mean())
print("Standard Deviation:", cv_scores.std())


# =====================================================
# 9. Cross-validation with MAE
# =====================================================
# sklearn negative MAE return করে, তাই minus দিয়ে positive করছি

cv_mae_scores = cross_val_score(
    model,
    X,
    y,
    cv=kf,
    scoring="neg_mean_absolute_error"
)

cv_mae_scores = -cv_mae_scores

print("\nCross-validation MAE scores:", cv_mae_scores)
print("Average CV MAE:", cv_mae_scores.mean())


# =====================================================
# 10. Normal train-test split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =====================================================
# 11. Train model
# =====================================================

model.fit(X_train, y_train)


# =====================================================
# 12. Test prediction
# =====================================================

y_pred = model.predict(X_test)


# =====================================================
# 13. Final evaluation
# =====================================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nFinal Test Result:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)


# =====================================================
# 14. New house prediction
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

print("\nPredicted House Price:", predicted_price[0])