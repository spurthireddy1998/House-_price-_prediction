import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="🏠 House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model_path = "house_price_model.pkl"

    if not os.path.exists(model_path):
        st.error("Model file 'house_price_model.pkl' not found.")
        st.stop()

    return joblib.load(model_path)

pipeline = load_model()

# -----------------------------
# Title
# -----------------------------
st.title("🏠 House Price Prediction")
st.write("Enter the house details below to estimate the house price.")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("About")
st.sidebar.info(
    """
    This application predicts house prices using a Machine Learning model
    built with Scikit-learn and deployed using Streamlit.
    """
)

# -----------------------------
# Input Form
# -----------------------------
with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input(
            "Area (sq ft)",
            min_value=100,
            value=1500,
            step=50
        )

        bedrooms = st.number_input(
            "Bedrooms",
            min_value=1,
            max_value=10,
            value=3
        )

        bathrooms = st.number_input(
            "Bathrooms",
            min_value=1,
            max_value=10,
            value=2
        )

        stories = st.number_input(
            "Stories",
            min_value=1,
            max_value=5,
            value=2
        )

        parking = st.number_input(
            "Parking",
            min_value=0,
            max_value=10,
            value=1
        )

    with col2:

        mainroad = st.selectbox(
            "Main Road",
            ["Yes", "No"]
        )

        guestroom = st.selectbox(
            "Guest Room",
            ["Yes", "No"]
        )

        basement = st.selectbox(
            "Basement",
            ["Yes", "No"]
        )

        hotwaterheating = st.selectbox(
            "Hot Water Heating",
            ["Yes", "No"]
        )

        airconditioning = st.selectbox(
            "Air Conditioning",
            ["Yes", "No"]
        )

        prefarea = st.selectbox(
            "Preferred Area",
            ["Yes", "No"]
        )

        furnishingstatus = st.selectbox(
            "Furnishing Status",
            [
                "Furnished",
                "Semi-furnished",
                "Unfurnished"
            ]
        )

    predict = st.form_submit_button("🔍 Predict House Price")

# -----------------------------
# Prediction
# -----------------------------
if predict:

    input_df = pd.DataFrame({
        "Area": [area],
        "Bedrooms": [bedrooms],
        "Bathrooms": [bathrooms],
        "Stories": [stories],
        "Parking": [parking],
        "MainRoad": [mainroad],
        "GuestRoom": [guestroom],
        "Basement": [basement],
        "HotWaterHeating": [hotwaterheating],
        "AirConditioning": [airconditioning],
        "PrefArea": [prefarea],
        "FurnishingStatus": [furnishingstatus]
    })

    st.subheader("Input Summary")
    st.dataframe(input_df, use_container_width=True)

    try:
        prediction = pipeline.predict(input_df)[0]

        st.success("Prediction Successful!")

        st.metric(
            label="Estimated House Price",
            value=f"₹ {prediction:,.2f}"
        )

        st.balloons()

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Developed using Streamlit • Scikit-learn • Python")
