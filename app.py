import streamlit as st
import pandas as pd
import pickle

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Car Price Predictor", layout="wide")

# --- CUSTOM CSS FOR 2X BIG WHITE BOLD METRICS ---
st.markdown("""
    <style>
    .centered-header { text-align: center; margin-bottom: 0px; }
    .centered-quote { text-align: center; color: #888; font-style: italic; margin-top: 0px; margin-bottom: 30px; }
    
    /* Price Result Box */
    .prediction-output {
        background-color: #1e3d33;
        color: #4ade80;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        border: 2px solid #2d5a4a;
        margin-top: 20px;
    }

    /* 2x Big White Bold Metrics */
    .metric-card {
        background-color: #0e1117;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        border: 1px solid #333;
    }
    .metric-label {
        color: #aaa;
        font-size: 18px;
        margin-bottom: 5px;
    }
    .metric-value {
        color: white;
        font-size: 40px; /* 2x Bigger */
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CENTERED HEADER & QUOTE ---
st.markdown("<h1 class='centered-header'>🚗 Smart Car Price Predictor 🚗</h1>", unsafe_allow_html=True)
st.markdown("<p class='centered-quote'>\"Analyzing patterns to estimate the market price accurately\"</p>", unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        with open('car_price_models.pkl', 'rb') as f:
            m = pickle.load(f)
        with open('ui_data.pkl', 'rb') as f:
            u = pickle.load(f)
        return m, u
    except:
        return None, None

model_assets, ui_data = load_assets()

# --- BRAND TO IMAGE MAPPING ---
brand_images = {
    "Audi": "https://i.pinimg.com/1200x/37/a0/af/37a0aff3f6788360533662a493087491.jpg",
    "BMW": "https://in.pinterest.com/pin/598064025558653376/",
    "Tesla": "https://in.pinterest.com/pin/8585055536115091/",
    "Mercedes": "https://in.pinterest.com/pin/734016439307968388/",
    "Toyota": "https://i.pinimg.com/1200x/30/c3/df/30c3dfa3554219b4c378affe63ac52ee.jpg",
    "Honda": "https://i.pinimg.com/1200x/ed/a0/bd/eda0bd911e4c0a16c6ff6a7507db2cb1.jpg"
    "Default": "https://i.pinimg.com/1200x/20/2d/3d/202d3dcca359714bb77606707880ab09.jpg"
    "Jeep": "https://i.pinimg.com/1200x/24/b5/2a/24b52a4c48da6105d3185c877131b85e.jpg"
    "Kia": "https://i.pinimg.com/1200x/d7/a4/16/d7a4164ca80ff793838ca6d7be22bf92.jpg"
    "MG": "https://i.pinimg.com/474x/c3/b1/33/c3b1336002d7b7e4b295d23a5b743ee8.jpg"
    "Tata": "https://i.pinimg.com/736x/fd/18/93/fd1893229b8e348bca4370be40f7e972.jpg"
    "Mahindra": "https://i.pinimg.com/736x/1a/eb/77/1aeb7730e12f4d7811c42f48609dc68e.jpg"
    "Skoda": "https://i.pinimg.com/736x/c7/9a/fa/c79afa9243a13261cf1f6e78abe0997b.jpg"
    "Toyota": "https://i.pinimg.com/1200x/5b/8d/b4/5b8db4bdb1a91e71d9a773b35d11585e.jpg"
    "Wolkswagen": "https://i.pinimg.com/1200x/85/be/40/85be40ffaf4c95522be65c18e242e986.jpg"
}

if model_assets and ui_data:
    # --- MAIN LAYOUT (TWO COLUMNS) ---
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("🛠️ Car Specifications")
        brand = st.selectbox("Select Car Brand", ui_data['brands'])
        year = st.slider("Registration Year", 2010, 2025, 2020)
        engine = st.number_input("Engine Size (Liters)", 0.5, 8.0, 2.0)
        fuel = st.selectbox("Fuel Type", ui_data['fuel_types'])
        trans = st.selectbox("Transmission", ui_data['transmissions'])
        mileage = st.number_input("Mileage (km)", 0, 500000, 40000)
        
        st.markdown("---")
        predict_btn = st.button("🚀 Calculate Predicted Price", use_container_width=True)

    with col2:
        # Dynamic Image based on Brand
        img_url = brand_images.get(brand, brand_images["Default"])
        st.image(img_url, use_container_width=True)
        
        if predict_btn:
            # Create input for model
            input_df = pd.DataFrame({
                'Car ID': [0], 'Brand': [brand], 'Year': [year],
                'Engine Size': [engine], 'Fuel Type': [fuel],
                'Transmission': [trans], 'Mileage': [mileage],
                'Condition': ['Used'], 'Model': ['Standard']
            })
            input_df = input_df[model_assets['feature_names']]
            prediction = model_assets['model'].predict(input_df)[0]
            
            st.markdown(f"<div class='prediction-output'>💰 Predicted Market Price: ₹ {prediction:,.2f}</div>", unsafe_allow_html=True)

    # --- PERFORMANCE METRICS (3 COLUMNS AT BOTTOM) ---
    st.markdown("---")
    st.markdown("<h3 style='text-align: center;'>📊 AI Model Performance</h3>", unsafe_allow_html=True)
    m_col1, m_col2, m_col3 = st.columns(3)

    with m_col1:
        st.markdown(f"""<div class='metric-card'><div class='metric-label'>Algorithm Used</div><div class='metric-value'>{model_assets['algo']}</div></div>""", unsafe_allow_html=True)
    
    with m_col2:
        st.markdown(f"""<div class='metric-card'><div class='metric-label'>Model Accuracy</div><div class='metric-value'>{model_assets['accuracy']}%</div></div>""", unsafe_allow_html=True)
    
    with m_col3:
        st.markdown(f"""<div class='metric-card'><div class='metric-label'>Mean Squared Error (MSE)</div><div class='metric-value'>{model_assets['mse']:,.0f}</div></div>""", unsafe_allow_html=True)

else:
    st.warning("⚠️ Files not found. Please run 'train_model.py' first.")




