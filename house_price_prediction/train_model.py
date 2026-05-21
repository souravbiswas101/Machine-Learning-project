# ============================================================
# House Price Prediction Project
# Dataset: Kaggle House Prices - Advanced Regression Techniques
# ============================================================

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


# ============================================================
# Optional XGBoost import
# ============================================================

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


# ============================================================
# 1. Folder paths
# ============================================================

DATA_DIR = "data"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(DATA_DIR, "sample_submission.csv")


# ============================================================
# 2. Check required files
# ============================================================

required_files = [
    TRAIN_PATH,
    TEST_PATH,
    SAMPLE_SUBMISSION_PATH
]

for file_path in required_files:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"{file_path} was not found. Please download train.csv, "
            f"test.csv, and sample_submission.csv from Kaggle and place "
            f"them inside the data folder."
        )


# ============================================================
# 3. Evaluation function
# ============================================================

def evaluate_model(y_true_log, y_pred_log):
    """
    The Kaggle House Prices competition uses a log-based error metric.
    In this project, SalePrice is transformed using log1p.
    Therefore, RMSE on the log target is used for validation.
    Lower RMSE is better.
    """

    rmse = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    mae = mean_absolute_error(y_true_log, y_pred_log)
    r2 = r2_score(y_true_log, y_pred_log)

    return rmse, mae, r2


# ============================================================
# 4. Version-safe OneHotEncoder function
# ============================================================

def make_onehot_encoder():
    """
    Different versions of scikit-learn use different parameters.

    New versions:
        sparse_output=False

    Older versions:
        sparse=False
    """

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# ============================================================
# 5. Load dataset
# ============================================================

print("Loading data...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

print("\nFirst 5 rows of training data:")
print(train_df.head())


# ============================================================
# 6. Basic data understanding
# ============================================================

print("\nDataset information:")
print(train_df.info())

print("\nTop 20 columns with missing values:")
print(train_df.isnull().sum().sort_values(ascending=False).head(20))

print("\nTarget column summary:")
print(train_df["SalePrice"].describe())


# ============================================================
# 7. Separate features and target
# ============================================================

test_ids = test_df["Id"]

X = train_df.drop(["SalePrice", "Id"], axis=1)
y = train_df["SalePrice"]

X_test_final = test_df.drop(["Id"], axis=1)

# SalePrice is usually skewed, so log transformation helps the model learn better.
y_log = np.log1p(y)


# ============================================================
# 8. Identify numerical and categorical columns
# ============================================================

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

print("\nNumber of numeric features:", len(numeric_features))
print("Number of categorical features:", len(categorical_features))


# ============================================================
# 9. Preprocessing
# ============================================================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", make_onehot_encoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    remainder="drop"
)


# ============================================================
# 10. Train-validation split
# ============================================================

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y_log,
    test_size=0.2,
    random_state=42
)

print("\nTraining data shape:", X_train.shape)
print("Validation data shape:", X_valid.shape)


# ============================================================
# 11. Define multiple machine learning models
# ============================================================

models = {
    "Linear Regression": LinearRegression(),

    "Ridge": Ridge(alpha=10),

    "Lasso": Lasso(
        alpha=0.001,
        random_state=42,
        max_iter=10000
    ),

    "ElasticNet": ElasticNet(
        alpha=0.001,
        l1_ratio=0.5,
        random_state=42,
        max_iter=10000
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),

    "Hist Gradient Boosting": HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=300,
        random_state=42
    ),

    "SVR": SVR(
        kernel="rbf",
        C=10,
        epsilon=0.05
    ),

    "KNN": KNeighborsRegressor(
        n_neighbors=5
    )
}

if XGBOOST_AVAILABLE:
    models["XGBoost"] = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
else:
    print("\nXGBoost is not available. Skipping XGBoost model.")


# ============================================================
# 12. Train and compare all models
# ============================================================

results = []
trained_pipelines = {}

print("\nTraining models...")

for model_name, model in models.items():

    print(f"\nTraining: {model_name}")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred_valid = pipeline.predict(X_valid)

    rmse, mae, r2 = evaluate_model(y_valid, y_pred_valid)

    results.append({
        "Model": model_name,
        "Validation_RMSE_Log": rmse,
        "Validation_MAE_Log": mae,
        "Validation_R2_Log": r2
    })

    trained_pipelines[model_name] = pipeline

    print(f"{model_name}: RMSE={rmse:.5f}, MAE={mae:.5f}, R2={r2:.5f}")


# ============================================================
# 13. Save model comparison results
# ============================================================

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Validation_RMSE_Log")

print("\nModel Comparison:")
print(results_df)

results_path = os.path.join(OUTPUT_DIR, "model_results.csv")
results_df.to_csv(results_path, index=False)

best_model_name = results_df.iloc[0]["Model"]
best_pipeline = trained_pipelines[best_model_name]

print("\nBest model before tuning:", best_model_name)


# ============================================================
# 14. Cross-validation for the best model
# ============================================================

print("\nRunning cross-validation for the best model...")

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    best_pipeline,
    X,
    y_log,
    cv=kf,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

cv_rmse_scores = -cv_scores

print("CV RMSE scores:", cv_rmse_scores)
print("Mean CV RMSE:", cv_rmse_scores.mean())


# ============================================================
# 15. Hyperparameter tuning
# ============================================================

print("\nHyperparameter tuning started...")

if best_model_name == "XGBoost" and XGBOOST_AVAILABLE:

    tuning_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    param_distributions = {
        "model__n_estimators": [300, 500, 800, 1000],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.08],
        "model__max_depth": [2, 3, 4, 5],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0]
    }

elif best_model_name in ["Random Forest", "Extra Trees"]:

    tuning_model = RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    )

    param_distributions = {
        "model__n_estimators": [200, 300, 500, 800],
        "model__max_depth": [None, 5, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", None]
    }

else:

    tuning_model = GradientBoostingRegressor(
        random_state=42
    )

    param_distributions = {
        "model__n_estimators": [200, 300, 500, 800],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.08],
        "model__max_depth": [2, 3, 4, 5],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__subsample": [0.7, 0.8, 0.9, 1.0]
    }


tuning_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", tuning_model)
])

random_search = RandomizedSearchCV(
    estimator=tuning_pipeline,
    param_distributions=param_distributions,
    n_iter=20,
    cv=5,
    scoring="neg_root_mean_squared_error",
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X, y_log)

print("\nBest parameters:")
print(random_search.best_params_)

print("\nBest CV RMSE:")
print(-random_search.best_score_)

final_model = random_search.best_estimator_


# ============================================================
# 16. Save final model
# ============================================================

model_save_path = os.path.join(MODEL_DIR, "best_house_price_model.pkl")
joblib.dump(final_model, model_save_path)

print("\nFinal model saved at:", model_save_path)


# ============================================================
# 17. Predict test data
# ============================================================

test_predictions_log = final_model.predict(X_test_final)

# Convert log price back to actual price
test_predictions = np.expm1(test_predictions_log)

# Avoid negative predictions
test_predictions = np.maximum(test_predictions, 0)


# ============================================================
# 18. Create Kaggle submission file
# ============================================================

submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": test_predictions
})

submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)

print("\nSubmission file created at:", submission_path)
print(submission.head())

print("\nProject completed successfully!")