import streamlit as st
import pandas as pd
import pickle
import os

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="House Rent Prediction",
    page_icon="🏠",
    layout="centered"
)

# ---------------------------------
# Safe Base Directory
# ---------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "rf_rent_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "encoders.pkl")

# ---------------------------------
# Load Model & Encoders
# ---------------------------------
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(ENCODER_PATH, "rb") as f:
        encoders = pickle.load(f)

except Exception as e:
    st.error("❌ Model files not loaded")
    st.exception(e)
    st.stop()

# ---------------------------------
# App Title
# ---------------------------------
st.title("🏠 House Rent Prediction")
st.caption("Random Forest Regression Model")
st.markdown("---")

# ---------------------------------
# User Inputs
# ---------------------------------
st.subheader("📋 Property Details")

bhk = st.slider("BHK", 1, 6, 2)
size = st.slider("House Size (sq.ft)", 300, 4000, 1000)
bathroom = st.slider("Number of Bathrooms", 1, 5, 2)

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

# ---------------------------------
# Prepare Input Data
# ---------------------------------
input_df = pd.DataFrame([{
    "BHK": bhk,
    "Size": size,
    "Bathroom": bathroom,
    "City": encoders["City"].transform([city])[0],
    "Furnishing Status": encoders["Furnishing Status"].transform([furnishing])[0],
    "Tenant Preferred": encoders["Tenant Preferred"].transform([tenant])[0]
}])

# ---------------------------------
# Prediction
# ---------------------------------
if st.button("Predict Rent 💰"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"🏷️ Estimated Monthly Rent: ₹ {prediction:,.0f}")
    except Exception as e:
        st.error("Prediction failed")
        st.exception(e)
