import streamlit as st
import pandas as pd
import pickle

# Page Config
st.set_page_config(page_title="AI Car Price Predictor", layout="wide")


# Load Brain & UI Data
@st.cache_resource
def load_assets():
    with open('car_price_models.pkl', 'rb') as f:
        m_assets = pickle.load(f)
    with open('ui_data.pkl', 'rb') as f:
        u_assets = pickle.load(f)
    return m_assets, u_assets


try:
    model_assets, ui_data = load_assets()

    st.title("🚗 Smart Car Price Predictor🚗")
    st.markdown("This model analyzes patterns to estimate the price accurately 🚘📈")

    # Hero Image
    st.image("https://images.unsplash.com/photo-1503376780353-7e6692767b70?auto=format&fit=crop&q=80&w=1200",
             caption="Advanced Machine Learning Prediction Engine")

    st.markdown("---")

    # Inputs
    col1, col2 = st.columns(2)
    with col1:
        brand = st.selectbox("Select Car Brand", ui_data['brands'])
        year = st.slider("Registration Year", 2010, 2025, 2020)
        engine = st.number_input("Engine Size (in Liters)", 0.5, 8.0, 2.0)

    with col2:
        fuel = st.selectbox("Fuel Type", ui_data['fuel_types'])
        trans = st.selectbox("Transmission Type", ui_data['transmissions'])
        mileage = st.number_input("Total Mileage (km)", 0, 500000, 30000)

    if st.button("🚀 Calculate Predicted Price", use_container_width=True):
        # Create input dataframe matching training features
        # We provide default values for 'Condition' and 'Model'
        input_data = pd.DataFrame({
            'Brand': [brand],
            'Year': [year],
            'Engine Size': [engine],
            'Fuel Type': [fuel],
            'Transmission': [trans],
            'Mileage': [mileage],
            'Condition': ['Used'],
            'Model': ['Standard']
        })

        # Reorder to match the model's expected feature order
        input_data = input_data[model_assets['feature_names']]

        # Predict
        prediction = model_assets['model'].predict(input_data)[0]

        # Results Display
        st.success(f"## 💰 Estimated Market Price: ₹ {prediction:,.2f}")

        st.markdown("---")
        st.subheader("📊 AI Model Performance")
        m1, m2, m3 = st.columns(3)
        m1.metric("Algorithm", model_assets['algo'])
        m2.metric("Accuracy", f"{model_assets['accuracy']}%")
        m3.metric("Mean Squared Error (MSE)", f"{model_assets['mse']:,.0f}")

except Exception as e:
    st.error(f"Required files not found or corrupted. Please run train_model.py first. Error: {e}")




