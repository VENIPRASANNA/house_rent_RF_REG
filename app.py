import streamlit as st
import pandas as pd
import pickle
import os

# Page config
st.set_page_config(
    page_title="House Rent Prediction",
    page_icon="🏠"
)

# Load model & encoders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(BASE_DIR, "rf_rent_model.pkl"), "rb"))
encoders = pickle.load(open(os.path.join(BASE_DIR, "encoders.pkl"), "rb"))

st.title("🏠 House Rent Prediction")
st.caption("Random Forest Regression Model")
st.markdown("---")

# Inputs
bhk = st.slider("BHK", 1, 6, 2)
size = st.slider("House Size (sq.ft)", 300, 4000, 1000)
bathroom = st.slider("Bathrooms", 1, 5, 2)

city = st.selectbox(
    "City",
    encoders["City"].classes_
)

furnishing = st.selectbox(
    "Furnishing Status",
    encoders["Furnishing Status"].classes_
)

tenant = st.selectbox(
    "Tenant Preferred",
    encoders["Tenant Preferred"].classes_
)

# Encode inputs
input_df = pd.DataFrame([{
    "BHK": bhk,
    "Size": size,
    "Bathroom": bathroom,
    "City": encoders["City"].transform([city])[0],
    "Furnishing Status": encoders["Furnishing Status"].transform([furnishing])[0],
    "Tenant Preferred": encoders["Tenant Preferred"].transform([tenant])[0]
}])

# Prediction
if st.button("Predict Rent"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Monthly Rent: ₹ {prediction:,.0f}")
