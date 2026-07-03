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
    /* Overall background */
    .stApp {
        background: linear-gradient(135deg, #1f1c2c 0%, #928dab 100%);
    }

    /* Title styling */
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        color: #ffffff;
        padding-bottom: 0px;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.4);
    }
    .sub-title {
        text-align: center;
        font-size: 16px;
        color: #f0f0f0;
        margin-top: -10px;
        margin-bottom: 25px;
    }

    /* Card container for results */
    .result-card {
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(10px);
        border-radius: 18px;
        padding: 25px;
        margin-top: 15px;
        border: 1px solid rgba(255,255,255,0.25);
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    }

    .price-tag {
        text-align: center;
        font-size: 40px;
        font-weight: 900;
        color: #00e676;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.3);
    }

    .info-row {
        display: flex;
        justify-content: space-between;
        color: #ffffff;
        font-size: 15px;
        padding: 4px 0;
        border-bottom: 1px solid rgba(255,255,255,0.15);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f2027, #203a43, #2c5364);
    }
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    /* Predict button */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #ff512f, #dd2476);
        color: white;
        font-weight: 700;
        font-size: 17px;
        padding: 12px 0;
        border-radius: 12px;
        border: none;
        transition: 0.3s;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    div.stButton > button:hover {
        transform: scale(1.03);
        box-shadow: 0 6px 20px rgba(0,0,0,0.4);
        background: linear-gradient(90deg, #dd2476, #ff512f);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------- HEADER ----------------------
st.markdown("<div class='main-title'>🚗 Car Price Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Get an instant estimated resale value for your car</div>", unsafe_allow_html=True)

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
st.sidebar.markdown("## 🔧 Car Details")
st.sidebar.markdown("Fill in the details below to get a price estimate.")
st.sidebar.markdown("---")

company = st.sidebar.selectbox('🏢 Select Company', companies)
names = sorted(df[df['company'] == company]['name'].unique())
name = st.sidebar.selectbox('🚘 Select Model', names)
year = st.sidebar.selectbox('📅 Purchase Year', years)
kms_driven = st.sidebar.number_input(
    '🛣️ Kilometers Driven',
    value=5000, min_value=1000, max_value=200000, step=1000
)
fuel_type = st.sidebar.radio('⛽ Fuel Type', ['Petrol', 'Diesel'], horizontal=True)

st.sidebar.markdown("---")
predict_btn = st.sidebar.button('🔮 Predict Price')

# ---------------------- MAIN CONTENT ----------------------
if not predict_btn:
    st.info("👈 Choose your car's details in the sidebar and click **Predict Price** to see the estimate.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Cars in Database", f"{len(df):,}")
    col2.metric("Companies Listed", f"{len(companies)}")
    col3.metric("Models Available", f"{df['name'].nunique()}")

else:
    with st.spinner('Crunching the numbers... 🔧'):
        time.sleep(0.8)

        myinput = pd.DataFrame(
            data=[[company, name, year, kms_driven, fuel_type]],
            columns=['company', 'name', 'year', 'kms_driven', 'fuel_type']
        )

        result = pipe.predict(myinput)
        predicted_price = float(np.array(result).flatten()[0])

    if predicted_price < 0:
        st.error('⚠️ Predicted price came out negative. Please double-check your inputs (e.g. year, kms driven).')
    else:
        st.balloons()
        st.markdown(f"""
        <div class="result-card">
            <div class="price-tag">₹ {round(predicted_price):,}</div>
            <p style="text-align:center; color:#dddddd; margin-top:-5px;">Estimated Resale Value</p>
            <hr style="border-color: rgba(255,255,255,0.2);">
            <div class="info-row"><span>🏢 Company</span><span>{company}</span></div>
            <div class="info-row"><span>🚘 Model</span><span>{name}</span></div>
            <div class="info-row"><span>📅 Year</span><span>{year}</span></div>
            <div class="info-row"><span>🛣️ Kms Driven</span><span>{kms_driven:,} km</span></div>
            <div class="info-row"><span>⛽ Fuel Type</span><span>{fuel_type}</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.caption("Note: This is a machine-learning based estimate and may vary from actual market price.")

# ---------------------- FOOTER ----------------------
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#eeeeee; font-size:13px;'>Made with ❤️ using Streamlit</p>",
    unsafe_allow_html=True
)
