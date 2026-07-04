import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="🏠 House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Prediction")
st.write("Enter the house details and click **Predict Price**.")

# -----------------------------
# Load Trained Pipeline
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("house_price_model.pkl")

pipeline = load_model()

# ===========================================================
# REPLACE THESE FEATURE NAMES WITH YOUR DATASET COLUMN NAMES
# (Exclude the target column 'Price')
# ===========================================================

FEATURES = {
    "Area": ("number", 1200),
    "Bedrooms": ("number", 3),
    "Bathrooms": ("number", 2),
    "Stories": ("number", 2),
    "Parking": ("number", 1),
    "MainRoad": ("text", "Yes"),
    "GuestRoom": ("text", "No"),
    "Basement": ("text", "No"),
    "HotWaterHeating": ("text", "No"),
    "AirConditioning": ("text", "Yes"),
    "PrefArea": ("text", "Yes"),
    "FurnishingStatus": ("text", "Semi-furnished")
}

user_input = {}

col1, col2 = st.columns(2)

i = 0
for feature, (dtype, default) in FEATURES.items():

    column = col1 if i % 2 == 0 else col2

    with column:
        if dtype == "number":
            value = st.number_input(feature, value=float(default))
        else:
            value = st.text_input(feature, value=str(default))

    user_input[feature] = value
    i += 1

input_df = pd.DataFrame([user_input])

st.subheader("Input Data")
st.dataframe(input_df)

if st.button("Predict House Price"):

    try:
        prediction = pipeline.predict(input_df)

        st.success("Prediction Successful!")

        st.metric(
            "Estimated House Price",
            f"₹ {prediction[0]:,.2f}"
        )

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)

st.markdown("---")
st.caption("Developed using Streamlit and Scikit-learn")