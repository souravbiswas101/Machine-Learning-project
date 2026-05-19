import streamlit as st
import pandas as pd
import joblib

# ==============================
# 1. Load saved model
# ==============================

model = joblib.load("california_house_price_model.pkl")


# ==============================
# 2. App title
# ==============================

st.title("California House Price Prediction App")

st.write("Enter house information below to predict the house price.")


# ==============================
# 3. User input
# ==============================

med_inc = st.number_input(
    "Median Income",
    min_value=0.0,
    max_value=20.0,
    value=5.0
)

house_age = st.number_input(
    "House Age",
    min_value=0.0,
    max_value=100.0,
    value=20.0
)

ave_rooms = st.number_input(
    "Average Rooms",
    min_value=0.0,
    max_value=20.0,
    value=5.0
)

ave_bedrms = st.number_input(
    "Average Bedrooms",
    min_value=0.0,
    max_value=10.0,
    value=1.0
)

population = st.number_input(
    "Population",
    min_value=0.0,
    max_value=40000.0,
    value=1000.0
)

ave_occup = st.number_input(
    "Average Occupancy",
    min_value=0.0,
    max_value=20.0,
    value=3.0
)

latitude = st.number_input(
    "Latitude",
    min_value=30.0,
    max_value=45.0,
    value=34.05
)

longitude = st.number_input(
    "Longitude",
    min_value=-125.0,
    max_value=-110.0,
    value=-118.25
)


# ==============================
# 4. Create input dataframe
# ==============================

new_house = pd.DataFrame({
    "MedInc": [med_inc],
    "HouseAge": [house_age],
    "AveRooms": [ave_rooms],
    "AveBedrms": [ave_bedrms],
    "Population": [population],
    "AveOccup": [ave_occup],
    "Latitude": [latitude],
    "Longitude": [longitude]
})


# ==============================
# 5. Prediction
# ==============================

if st.button("Predict House Price"):
    prediction = model.predict(new_house)[0]

    price_dollar = prediction * 100000

    st.success(f"Predicted House Value: {prediction:.2f}")
    st.success(f"Predicted House Price: ${price_dollar:,.2f}")