import streamlit as st
import pandas as pd
import pickle

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Car Price Predictor", layout="wide")

# --- CUSTOM CSS FOR CENTERING ---
st.markdown("""
    <style>
    .centered-text { text-align: center; }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CENTERED HEADER & QUOTE ---
st.markdown("<h1 class='centered-text'>🚗 Smart Car Price Predictor 🚗</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='centered-text'>\"Predicting Market Value with Precision and AI\"</h3>", unsafe_allow_html=True)
st.markdown("---")

# --- LOAD DATA (For selection options) ---
try:
    df = pd.read_csv('DATASET CAR PRICE.csv')
except:
    st.error("Error: 'DATASET CAR PRICE.csv' not found. Please ensure the file is in the same folder.")
    st.stop()

# --- BRAND TO IMAGE MAPPING ---
# These are high-quality placeholder images. You can replace these URLs with your own.
brand_images = {
    "Audi": "https://images.unsplash.com/photo-1541348263662-e0c86433ec1e?q=80&w=800",
    "BMW": "https://images.unsplash.com/photo-1555215695-3004980ad54e?q=80&w=800",
    "Tesla": "https://images.unsplash.com/photo-1560958089-b8a1929cea89?q=80&w=800",
    "Mercedes": "https://images.unsplash.com/photo-1503376780353-7e6692767b70?q=80&w=800",
    "Toyota": "https://images.unsplash.com/photo-1621007947382-bb3c3994e3fb?q=80&w=800",
    "Hyundai": "https://images.unsplash.com/photo-1619682817481-e994891cd1f5?q=80&w=800",
    "Default": "https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?q=80&w=800"
}

# --- TWO COLUMN LAYOUT ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("🛠️ Select Car Features")
    
    brand = st.selectbox("Select Car Brand", sorted(df['Brand'].unique()))
    model = st.selectbox("Select Model", sorted(df[df['Brand'] == brand]['Model'].unique()))
    year = st.slider("Registration Year", 2010, 2025, 2020)
    engine_size = st.number_input("Engine Size (in Liters)", 0.5, 10.0, 2.0)
    fuel_type = st.selectbox("Fuel Type", df['Fuel Type'].unique())
    transmission = st.selectbox("Transmission", df['Transmission'].unique())
    mileage = st.number_input("Total Mileage (km)", 0, 500000, 50000)

with col2:
    st.subheader("🏎️ Car Visualization")
    
    # Display Image Based on Brand
    image_url = brand_images.get(brand, brand_images["Default"])
    st.image(image_url, caption=f"Selected Brand: {brand}", use_container_width=True)
    
    st.markdown("---")
    
    # Prediction Button
    if st.button("Calculate Predicted Price", use_container_width=True):
        # Here you would load your pickle model:
        # model = pickle.load(open('car_price_model.pkl', 'rb'))
        # result = model.predict([[...]])
        
        # Placeholder for visual result
        st.markdown("<div class='prediction-box'>Estimated Market Price: $24,500.00</div>", unsafe_allow_html=True)
        st.balloons()

# --- FOOTER ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Project Credits: Student Intern | BKMN University</p>", unsafe_allow_html=True)
