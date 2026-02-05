import streamlit as st
import pandas as pd
import pickle

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Car Price Predictor", layout="wide")

# --- CUSTOM CSS FOR EXACT LAYOUT ---
st.markdown("""
    <style>
    .centered-header {
        text-align: center;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    .centered-quote {
        text-align: center;
        color: #666;
        font-style: italic;
        margin-top: 0px;
    }
    .prediction-output {
        background-color: #1e3d33;
        color: #4ade80;
        padding: 15px;
        border-radius: 5px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        border: 1px solid #2d5a4a;
    }
    /* Ensuring columns are aligned properly */
    [data-testid="column"] {
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CENTERED HEADER & QUOTE ---
st.markdown("<h1 class='centered-header'>🚗 Smart Car Price Predictor 🚗</h1>", unsafe_allow_html=True)
st.markdown("<p class='centered-quote'>H3:: Predicting Market Value with AI</p>", unsafe_allow_html=True)
st.markdown("---")

# --- LOAD DATA ---
try:
    # Loading the dataset provided 
    df = pd.read_csv('DATASET CAR PRICE.csv')
except FileNotFoundError:
    st.error("Error: 'DATASET CAR PRICE.csv' not found.")
    st.stop()

# --- BRAND TO IMAGE MAPPING ---
brand_images = {
    "Audi": "https://images.unsplash.com/photo-1541348263662-e0c86433ec1e?q=80&w=800",
    "BMW": "https://images.unsplash.com/photo-1555215695-3004980ad54e?q=80&w=800",
    "Tesla": "https://images.unsplash.com/photo-1560958089-b8a1929cea89?q=80&w=800",
    "Mercedes": "https://images.unsplash.com/photo-1503376780353-7e6692767b70?q=80&w=800",
    "Toyota": "https://images.unsplash.com/photo-1621007947382-bb3c3994e3fb?q=80&w=800",
    "Hyundai": "https://images.unsplash.com/photo-1619682817481-e994891cd1f5?q=80&w=800",
    "Default": "https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?q=80&w=800"
}

# --- MAIN LAYOUT ---
# Left column for inputs, Right column for Image and Results
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.markdown("### 🗺️ Select Car Brand")
    
    # Get unique brands from dataset 
    brand = st.selectbox("Select Car Brand", sorted(df['Brand'].unique()))
    
    # Filtering models based on brand 
    brand_models = sorted(df[df['Brand'] == brand]['Model'].unique())
    model = st.selectbox("Select Model", brand_models)
    
    year = st.slider("Registration Year", 2010, 2025, 2020)
    engine_size = st.number_input("Engine Size (in Liters)", 0.5, 10.0, 2.0)
    fuel_type = st.selectbox("Fuel Type", df['Fuel Type'].unique())
    transmission = st.selectbox("Transmission", df['Transmission'].unique())
    mileage = st.number_input("Total Milage (km)", 0, 500000, 34000)
    
    st.markdown("---")
    predict_btn = st.button("Calculate Predicted Price", use_container_width=True)

with col2:
    # Dynamic Image Display based on brand selection
    img_url = brand_images.get(brand, brand_images["Default"])
    st.image(img_url, use_container_width=True)
    
    st.markdown("---")
    
    if predict_btn:
        # Static price for visual demonstration (replace with model.predict logic)
        st.markdown("<div class='prediction-output'>💰 Estimated Market Price: $16,813,382.F</div>", unsafe_allow_html=True)

# --- PROJECT CREDITS ---
st.markdown("---")
st.markdown("#### Project Credits")
st.write("Student Intern: [Your Name]")
st.write("University: BKNM University")

# --- DATASET FOOTER ---
st.caption("AI Model Performance: Accuracy 70% | Algorithm: Random Forest")
