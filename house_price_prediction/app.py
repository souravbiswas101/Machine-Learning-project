# ============================================================
# Streamlit App for House Price Prediction
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# Page setup
# ============================================================

st.set_page_config(
    page_title="House Price Prediction App",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Prediction App")
st.write("Upload a Kaggle test-style CSV file and predict house prices.")


# ============================================================
# Model path
# ============================================================

MODEL_PATH = os.path.join("models", "best_house_price_model.pkl")


# ============================================================
# Check model file
# ============================================================

if not os.path.exists(MODEL_PATH):
    st.error("Model file was not found.")
    st.write("Please train the model first by running:")
    st.code("python train_model.py")
    st.stop()


# ============================================================
# Load trained model
# ============================================================

model = joblib.load(MODEL_PATH)

st.success("Model loaded successfully!")


# ============================================================
# CSV upload
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Kaggle test.csv file",
    type=["csv"]
)


# ============================================================
# Prediction section
# ============================================================

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    original_df = df.copy()

    # Remove Id column before prediction if it exists
    if "Id" in df.columns:
        ids = df["Id"]
        X_input = df.drop("Id", axis=1)
    else:
        ids = None
        X_input = df.copy()

    # Remove SalePrice column if it exists by mistake
    if "SalePrice" in X_input.columns:
        X_input = X_input.drop("SalePrice", axis=1)

    st.write("Total rows:", len(df))
    st.write("Total columns:", len(df.columns))

    if st.button("Predict House Price"):

        try:
            predictions_log = model.predict(X_input)

            # Convert log prediction back to actual price
            predictions = np.expm1(predictions_log)

            # Avoid negative predictions
            predictions = np.maximum(predictions, 0)

            result_df = original_df.copy()
            result_df["Predicted_SalePrice"] = predictions

            st.subheader("Prediction Result")
            st.dataframe(result_df.head(20))

            csv = result_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Prediction CSV",
                data=csv,
                file_name="house_price_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error("Prediction failed.")
            st.write("Possible reason: The uploaded CSV columns do not match the training dataset columns.")
            st.write("Error details:")
            st.code(str(e))

else:
    st.info("Please upload a CSV file to start prediction.")