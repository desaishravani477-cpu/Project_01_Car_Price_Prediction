#streamlit run app.py
import streamlit as st
import pickle as pkl
import numpy as np
import pandas as pd
import time

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------------------- CUSTOM CSS ----------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Roboto', 'Segoe UI', sans-serif;
    }

    /* Overall background - clean white/light gray */
    .stApp {
        background-color: #f8f9fa;
    }

    /* Header bar */
    .header-bar {
        background-color: #ffffff;
        border-bottom: 1px solid #e0e0e0;
        padding: 18px 0 16px 0;
        margin-bottom: 25px;
        text-align: center;
    }
    .main-title {
        font-size: 28px;
        font-weight: 700;
        color: #202124;
        margin: 0;
    }
    .main-title span {
        color: #1a73e8;
    }
    .sub-title {
        font-size: 14px;
        color: #5f6368;
        margin-top: 4px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    section[data-testid="stSidebar"] * {
        color: #202124 !important;
    }
    .sidebar-heading {
        font-size: 15px;
        font-weight: 700;
        color: #202124 !important;
        margin-bottom: 2px;
    }
    .sidebar-caption {
        font-size: 12px;
        color: #5f6368 !important;
        margin-bottom: 14px;
    }

    /* Input fields */
    div[data-baseweb="select"] > div,
    .stNumberInput input,
    .stTextInput input {
        background-color: #ffffff !important;
        color: #202124 !important;
        border: 1px solid #dadce0 !important;
        border-radius: 6px !important;
    }
    div[data-baseweb="select"] span {
        color: #202124 !important;
    }
    ul[data-testid="stVirtualDropdown"] li {
        background-color: #ffffff !important;
        color: #202124 !important;
    }
    ul[data-testid="stVirtualDropdown"] li:hover {
        background-color: #e8f0fe !important;
    }

    /* Radio buttons accent */
    input[type="radio"] {
        accent-color: #1a73e8 !important;
    }

    /* Predict button - Google style solid blue */
    div.stButton > button {
        width: 100%;
        background-color: #1a73e8;
        color: #ffffff;
        font-weight: 500;
        font-size: 14px;
        letter-spacing: 0.3px;
        padding: 10px 8px;
        border-radius: 6px;
        border: none;
        white-space: normal;
        height: auto;
        line-height: 1.4;
        transition: background-color 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 1px 2px rgba(60,64,67,0.3);
    }
    div.stButton > button:hover {
        background-color: #1765cc;
        box-shadow: 0 1px 3px rgba(60,64,67,0.45);
    }
    div.stButton > button p {
        font-size: 14px !important;
        white-space: normal !important;
    }

    /* Result card */
    .result-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 24px 28px;
        margin-top: 10px;
        box-shadow: 0 1px 3px rgba(60,64,67,0.15);
    }
    .price-label {
        font-size: 13px;
        color: #5f6368;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .price-tag {
        font-size: 36px;
        font-weight: 700;
        color: #188038;
        margin: 4px 0 16px 0;
    }
    .info-row {
        display: flex;
        justify-content: space-between;
        color: #3c4043;
        font-size: 14px;
        padding: 8px 0;
        border-top: 1px solid #f1f3f4;
    }
    .info-row span:first-child {
        color: #5f6368;
    }
    .info-row span:last-child {
        font-weight: 500;
    }

    /* Metric cards on landing view */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 14px 10px;
        box-shadow: 0 1px 2px rgba(60,64,67,0.1);
    }

    /* Footer */
    .footer {
        text-align: center;
        font-size: 12px;
        color: #80868b;
        padding: 20px 0 6px 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------- HEADER ----------------------
st.markdown("""
<div class="header-bar">
    <div class="main-title">🚗 <span>Car</span>Price Predictor</div>
    <div class="sub-title">Get an instant, data-driven estimate of your car's resale value</div>
</div>
""", unsafe_allow_html=True)

# ---------------------- LOAD DATA/MODEL ----------------------
@st.cache_resource
def load_model():
    return pkl.load(open('CPP.pkl', 'rb'))

@st.cache_data
def load_data():
    return pd.read_csv('final_data.csv')

pipe = load_model()
df = load_data()

companies = sorted(df['company'].unique())
years = list(range(2027, 1999, -1))

# ---------------------- SIDEBAR INPUTS ----------------------
st.sidebar.markdown('<div class="sidebar-heading">Car Details</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-caption">Fill in the details to get a price estimate</div>', unsafe_allow_html=True)

company = st.sidebar.selectbox('Company', companies)
names = sorted(df[df['company'] == company]['name'].unique())
name = st.sidebar.selectbox('Model', names)
year = st.sidebar.selectbox('Purchase Year', years)
kms_driven = st.sidebar.number_input(
    'Kilometers Driven',
    value=5000, min_value=1000, max_value=200000, step=1000
)
fuel_type = st.sidebar.radio('Fuel Type', ['Petrol', 'Diesel'], horizontal=True)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.sidebar.button('Predict Price')

# ---------------------- MAIN CONTENT ----------------------
if not predict_btn:
    st.info("Choose your car's details in the sidebar, then click **Predict Price** to see the estimate.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Cars in Database", f"{len(df):,}")
    col2.metric("Companies Listed", f"{len(companies)}")
    col3.metric("Models Available", f"{df['name'].nunique()}")

else:
    with st.spinner('Calculating estimate...'):
        time.sleep(0.6)

        myinput = pd.DataFrame(
            data=[[company, name, year, kms_driven, fuel_type]],
            columns=['company', 'name', 'year', 'kms_driven', 'fuel_type']
        )

        result = pipe.predict(myinput)
        predicted_price = float(np.array(result).flatten()[0])

    if predicted_price < 0:
        st.error('Predicted price came out negative. Please double-check your inputs (e.g. year, kms driven).')
    else:
        st.markdown(f"""
        <div class="result-card">
            <div class="price-label">Estimated Resale Value</div>
            <div class="price-tag">₹ {round(predicted_price):,}</div>
            <div class="info-row"><span>Company</span><span>{company}</span></div>
            <div class="info-row"><span>Model</span><span>{name}</span></div>
            <div class="info-row"><span>Year</span><span>{year}</span></div>
            <div class="info-row"><span>Kms Driven</span><span>{kms_driven:,} km</span></div>
            <div class="info-row"><span>Fuel Type</span><span>{fuel_type}</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.caption("This is a machine-learning based estimate and may vary from actual market price.")

# ---------------------- FOOTER ----------------------
st.markdown('<div class="footer">Built with Streamlit</div>', unsafe_allow_html=True)
