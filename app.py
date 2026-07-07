"""
Car Price Prediction — Premium ML Web App
Developed by Shravani Desai

Built with Streamlit, Plotly and a trained ML pipeline (CPP.pkl).
The prediction logic itself is untouched — only UI/UX and extra
features (charts, history, PDF export, theming) have been added.
"""

import io
import pickle
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

# =============================================================
# PAGE CONFIGURATION
# =============================================================

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon=":material/directions_car:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================
# SESSION STATE INITIALISATION
# =============================================================

def init_session_state():
    """Set up default values in session_state on first run."""
    defaults = {
        "theme": "light",
        "history": [],
        "last_prediction": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

# =============================================================
# THEME DEFINITIONS
# =============================================================

LIGHT_THEME = {
    "bg": "#F5F7FA",
    "bg_gradient": "linear-gradient(135deg, #F5F7FA 0%, #E4ECFB 100%)",
    "card_bg": "rgba(255, 255, 255, 0.75)",
    "card_border": "rgba(255, 255, 255, 0.5)",
    "text": "#12151C",
    "muted_text": "#5A6472",
    "primary": "#0B5FFF",
    "primary_dark": "#0A3FBF",
    "accent": "#111827",
    "sidebar_bg": "#FFFFFF",
    "sidebar_text": "#12151C",
    "sidebar_border": "#E3E7EE",
    "shadow": "0px 10px 30px rgba(17, 24, 39, 0.08)",
    "success": "#0F9D58",
    "warning": "#E8A200",
    "danger": "#E5484D",
}

DARK_THEME = {
    "bg": "#0B0D12",
    "bg_gradient": "linear-gradient(135deg, #0B0D12 0%, #12161F 100%)",
    "card_bg": "rgba(255, 255, 255, 0.06)",
    "card_border": "rgba(255, 255, 255, 0.08)",
    "text": "#F2F4F8",
    "muted_text": "#9AA4B2",
    "primary": "#4C8DFF",
    "primary_dark": "#2E6BE0",
    "accent": "#FFFFFF",
    "sidebar_bg": "#FFFFFF",
    "sidebar_text": "#12151C",
    "sidebar_border": "#E3E7EE",
    "shadow": "0px 10px 30px rgba(0, 0, 0, 0.5)",
    "success": "#34C77B",
    "warning": "#F2B705",
    "danger": "#FF6B6B",
}


def get_theme():
    """Return the active theme color dictionary."""
    return DARK_THEME if st.session_state.theme == "dark" else LIGHT_THEME


# =============================================================
# CUSTOM CSS INJECTION
# =============================================================

def inject_css(t):
    """Inject custom CSS built from the active theme dictionary."""
    st.markdown(f"""
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    .stApp {{
        background: {t['bg_gradient']};
        color: {t['text']};
    }}

    /* ---------- Header ---------- */
    .app-title {{
        text-align: center;
        font-size: 46px;
        font-weight: 800;
        letter-spacing: -1px;
        color: {t['text']};
        margin-bottom: 0px;
    }}

    .app-subtitle {{
        text-align: center;
        font-size: 17px;
        font-weight: 400;
        color: {t['muted_text']};
        margin-top: 6px;
        margin-bottom: 30px;
    }}

    /* ---------- Glass Card ---------- */
    .glass-card {{
        background: {t['card_bg']};
        border: 1px solid {t['card_border']};
        border-radius: 20px;
        padding: 28px;
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        box-shadow: {t['shadow']};
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        margin-bottom: 24px;
    }}

    .glass-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0px 18px 40px rgba(0,0,0,0.15);
    }}

    .section-heading {{
        font-size: 22px;
        font-weight: 700;
        color: {t['text']};
        margin-bottom: 14px;
        letter-spacing: -0.3px;
    }}

    /* ---------- Result Card ---------- */
    .result-card {{
        background: linear-gradient(135deg, {t['primary']} 0%, {t['primary_dark']} 100%);
        border-radius: 22px;
        padding: 34px;
        text-align: center;
        box-shadow: 0px 14px 34px rgba(11, 95, 255, 0.35);
        color: white;
        margin-bottom: 24px;
    }}

    .result-label {{
        font-size: 16px;
        font-weight: 500;
        opacity: 0.85;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }}

    .result-value {{
        font-size: 48px;
        font-weight: 800;
        margin-top: 8px;
        letter-spacing: -1px;
    }}

    /* ---------- Advice Badge ---------- */
    .advice-badge {{
        display: inline-block;
        padding: 8px 20px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 14px;
        letter-spacing: 0.4px;
        text-transform: uppercase;
        margin-top: 14px;
    }}

    /* ---------- Buttons ---------- */
    .stButton>button {{
        background: {t['primary']};
        color: white;
        border: none;
        border-radius: 12px;
        height: 52px;
        width: 100%;
        font-size: 16px;
        font-weight: 600;
        letter-spacing: 0.2px;
        transition: all 0.25s ease;
        box-shadow: 0px 6px 16px rgba(11, 95, 255, 0.25);
    }}

    .stButton>button:hover {{
        background: {t['primary_dark']};
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0px 10px 24px rgba(11, 95, 255, 0.35);
    }}

    .stDownloadButton>button {{
        background: {t['accent']};
        color: white;
        border-radius: 12px;
        font-weight: 600;
        height: 48px;
        width: 100%;
        border: none;
        transition: 0.25s ease;
    }}

    .stDownloadButton>button:hover {{
        transform: translateY(-2px);
        opacity: 0.9;
    }}

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {{
        background: {t['sidebar_bg']};
        border-right: 1px solid {t['sidebar_border']};
    }}

    section[data-testid="stSidebar"] * {{
        color: {t['sidebar_text']} !important;
    }}

    section[data-testid="stSidebar"] .stButton>button {{
        background: {t['primary']};
        color: white !important;
    }}

    /* ---------- Inputs ---------- */

    /* The closed select box shown in the sidebar */
    div[data-baseweb="select"] > div {{
        border-radius: 12px !important;
        border: 1px solid #D7DCE5 !important;
        background-color: #FFFFFF !important;
    }}

    div[data-baseweb="select"] > div * {{
        color: #12151C !important;
        fill: #12151C !important;
    }}

    /* The dropdown option list (rendered in a floating popover) */
    ul[data-baseweb="menu"] {{
        background-color: #FFFFFF !important;
        border: 1px solid #E3E7EE !important;
        border-radius: 12px !important;
    }}

    ul[data-baseweb="menu"] li {{
        color: #12151C !important;
        background-color: #FFFFFF !important;
    }}

    ul[data-baseweb="menu"] li:hover {{
        background-color: #EEF2FA !important;
    }}

    /* Number input box */
    .stNumberInput input {{
        border-radius: 12px !important;
        background-color: #FFFFFF !important;
        color: #12151C !important;
        border: 1px solid #D7DCE5 !important;
    }}

    /* ---------- Table ---------- */
    .stDataFrame {{
        border-radius: 16px;
        overflow: hidden;
    }}

    /* ---------- Footer ---------- */
    .footer {{
        text-align: center;
        padding-top: 40px;
        padding-bottom: 20px;
        font-size: 14px;
        color: {t['muted_text']};
        border-top: 1px solid {t['card_border']};
        margin-top: 50px;
    }}

    .footer b {{
        color: {t['text']};
    }}

    </style>
    """, unsafe_allow_html=True)


# =============================================================
# DATA / MODEL LOADING
# =============================================================

@st.cache_resource(show_spinner=False)
def load_model(path="CPP.pkl"):
    """Load the trained ML pipeline. Returns None if unavailable."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner=False)
def load_data(path="final_data.csv"):
    """Load the dataset used to populate dropdowns and charts."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None


pipe = load_model()
df = load_data()

# =============================================================
# SIDEBAR THEME TOGGLE
# =============================================================

with st.sidebar:
    st.markdown("### Appearance")
    theme_choice = st.toggle(
        "Dark Mode",
        value=(st.session_state.theme == "dark"),
        key="theme_toggle",
    )
    st.session_state.theme = "dark" if theme_choice else "light"
    st.divider()

theme = get_theme()
inject_css(theme)

# =============================================================
# HEADER
# =============================================================

st.markdown('<div class="app-title">Car Price Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Estimate the fair market value of a used car using a trained machine learning model</div>',
    unsafe_allow_html=True,
)

# Stop early with a clear message if required files are missing.
if df is None or pipe is None:
    st.error(
        "Required files were not found. Please make sure **CPP.pkl** and "
        "**final_data.csv** are placed in the same folder as this app."
    )
    st.stop()


# =============================================================
# SIDEBAR — CAR DETAIL INPUTS
# =============================================================

st.sidebar.header("Enter Car Details")

companies = sorted(df["company"].unique())
company = st.sidebar.selectbox("Company", companies)

names = sorted(df[df["company"] == company]["name"].unique())
name = st.sidebar.selectbox("Model", names)

years = list(range(2000, 2027))
year = st.sidebar.selectbox("Manufacturing Year", years, index=len(years) - 5)

km_driven = st.sidebar.number_input(
    "Kilometers Driven", min_value=1000, max_value=200000, value=50000, step=1000
)

fuel = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel"])

st.sidebar.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.sidebar.button("Predict Price")

st.sidebar.divider()
if st.sidebar.button("Clear Prediction History"):
    st.session_state.history = []
    st.sidebar.success("History cleared.")

# =============================================================
# BUYING ADVICE LOGIC
# =============================================================

def get_buying_advice(predicted_price: float, company_name: str) -> tuple[str, str]:
    """
    Compare the predicted price to the average price of the same
    company in the dataset (as a rough market reference) and return
    an advice label plus a theme color key.
    """
    # 'Selling_Price' style columns vary by dataset; try common names.
    price_col = None
    for candidate in ["Price", "price", "selling_price", "Selling_Price"]:
        if candidate in df.columns:
            price_col = candidate
            break

    if price_col is None:
        return "Fair Price", "warning"

    company_avg = df[df["company"] == company_name][price_col].mean()

    if pd.isna(company_avg) or company_avg == 0:
        return "Fair Price", "warning"

    ratio = predicted_price / company_avg

    if ratio <= 0.9:
        return "Good Deal", "success"
    elif ratio <= 1.15:
        return "Fair Price", "warning"
    else:
        return "Premium Price", "danger"


# =============================================================
# PDF REPORT GENERATION
# =============================================================

def generate_pdf_report(details: dict) -> bytes:
    """Build a downloadable PDF summarising the prediction."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"], fontSize=22, textColor=colors.HexColor("#0B5FFF")
    )
    subtitle_style = ParagraphStyle(
        "SubtitleStyle", parent=styles["Normal"], fontSize=11, textColor=colors.grey
    )

    elements = [
        Paragraph("Car Price Prediction Report", title_style),
        Paragraph(f"Generated on {datetime.now().strftime('%d %B %Y, %I:%M %p')}", subtitle_style),
        Spacer(1, 20),
    ]

    table_data = [
        ["Field", "Value"],
        ["Company", details["company"]],
        ["Model", details["name"]],
        ["Manufacturing Year", str(details["year"])],
        ["Kilometers Driven", f"{details['km_driven']:,} km"],
        ["Fuel Type", details["fuel"]],
        ["Predicted Price", f"Rs. {details['price']:,.0f}"],
        ["AI Buying Advice", details["advice"]],
    ]

    table = Table(table_data, colWidths=[6 * cm, 8 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B5FFF")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10.5),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
        ("TOPPADDING", (0, 0), (-1, 0), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7FA")]),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 30))
    elements.append(Paragraph(
        "This report was generated automatically by a machine learning model "
        "trained on historical used-car listings. Estimated prices are "
        "indicative and may vary from actual market value.",
        styles["Normal"],
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


# =============================================================
# MAIN LAYOUT — SELECTED DETAILS + LOGO
# =============================================================

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Selected Vehicle Details</div>', unsafe_allow_html=True)
detail_cols = st.columns(2)
with detail_cols[0]:
    st.markdown(f"**Company:** {company}")
    st.markdown(f"**Model:** {name}")
    st.markdown(f"**Year:** {year}")
with detail_cols[1]:
    st.markdown(f"**Kilometers Driven:** {km_driven:,} km")
    st.markdown(f"**Fuel Type:** {fuel}")
st.markdown('</div>', unsafe_allow_html=True)

# =============================================================
# PREDICTION
# =============================================================

if predict_clicked:

    progress_text = st.empty()
    progress_bar = st.progress(0)

    steps = [
        (20, "Validating inputs..."),
        (50, "Running machine learning model..."),
        (80, "Computing market comparison..."),
        (100, "Finalising results..."),
    ]
    for pct, msg in steps:
        progress_text.markdown(f"**{msg}**")
        progress_bar.progress(pct)
        import time
        time.sleep(0.4)

    progress_text.empty()
    progress_bar.empty()

    # ---- Exact prediction logic (unchanged) ----
    myinput = [[company, name, year, km_driven, fuel]]
    columns = ["company", "name", "year", "kms_driven", "fuel_type"]
    myinput = pd.DataFrame(myinput, columns=columns)
    result = pipe.predict(myinput)
    price = float(result[0,0])
    # ---------------------------------------------

    if price < 0:
        st.error("Prediction failed. Please try different input values.")
    else:
        advice, advice_key = get_buying_advice(price, company)
        badge_color = theme[advice_key]

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Estimated Price</div>
            <div class="result-value">Rs. {price:,.0f}</div>
            <div class="advice-badge" style="background:{badge_color}; color:white;">
                {advice}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.success("Prediction completed successfully.")

        # Save to history
        record = {
            "Company": company,
            "Model": name,
            "Year": year,
            "KM Driven": km_driven,
            "Fuel": fuel,
            "Predicted Price": round(price, 0),
            "Advice": advice,
            "Timestamp": datetime.now().strftime("%d-%m-%Y %H:%M"),
        }
        st.session_state.history.append(record)
        st.session_state.last_prediction = {
            "company": company, "name": name, "year": year,
            "km_driven": km_driven, "fuel": fuel, "price": price, "advice": advice,
        }

        # PDF download
        pdf_bytes = generate_pdf_report(st.session_state.last_prediction)
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name=f"car_price_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
        )

# =============================================================
# PREDICTION HISTORY
# =============================================================

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Prediction History</div>', unsafe_allow_html=True)

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True, hide_index=True)
else:
    st.markdown("No predictions made yet in this session.")

st.markdown('</div>', unsafe_allow_html=True)

# =============================================================
# ANALYTICS / CHARTS
# =============================================================

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Market Analytics</div>', unsafe_allow_html=True)

price_col = next((c for c in ["Price", "price", "selling_price", "Selling_Price"] if c in df.columns), None)

plot_template = "plotly_dark" if st.session_state.theme == "dark" else "plotly_white"

if price_col:
    chart_tabs = st.tabs([
        "Price Distribution",
        "Depreciation by Year",
        "Company-wise Average Price",
        "Fuel Type Distribution",
        "Year-wise Average Price",
    ])

    with chart_tabs[0]:
        fig = px.histogram(
            df, x=price_col, nbins=40, template=plot_template,
            color_discrete_sequence=[theme["primary"]],
        )
        fig.update_layout(title="Price Distribution Across Dataset", xaxis_title="Price", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with chart_tabs[1]:
        if "year" in df.columns:
            dep = df.groupby("year")[price_col].mean().reset_index()
            fig = px.line(
                dep, x="year", y=price_col, markers=True, template=plot_template,
                color_discrete_sequence=[theme["primary"]],
            )
            fig.update_layout(title="Average Price by Manufacturing Year (Depreciation Trend)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Year column not found in dataset.")

    with chart_tabs[2]:
        comp_avg = df.groupby("company")[price_col].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(
            comp_avg, x="company", y=price_col, template=plot_template,
            color=price_col, color_continuous_scale="Blues",
        )
        fig.update_layout(title="Company-wise Average Price", xaxis_title="Company", yaxis_title="Average Price")
        st.plotly_chart(fig, use_container_width=True)

    with chart_tabs[3]:
        if "fuel_type" in df.columns:
            fuel_counts = df["fuel_type"].value_counts().reset_index()
            fuel_counts.columns = ["fuel_type", "count"]
            fig = px.pie(
                fuel_counts, names="fuel_type", values="count", template=plot_template,
                color_discrete_sequence=px.colors.sequential.Blues_r,
            )
            fig.update_layout(title="Fuel Type Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Fuel type column not found in dataset.")

    with chart_tabs[4]:
        if "year" in df.columns:
            yearly = df.groupby("year")[price_col].mean().reset_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=yearly["year"], y=yearly[price_col],
                mode="lines+markers", fill="tozeroy",
                line=dict(color=theme["primary"], width=3),
            ))
            fig.update_layout(
                title="Year-wise Average Price", template=plot_template,
                xaxis_title="Year", yaxis_title="Average Price",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Year column not found in dataset.")
else:
    st.warning(
        "No recognizable price column (e.g. 'Price', 'selling_price') was found "
        "in final_data.csv, so market analytics charts cannot be generated."
    )

st.markdown('</div>', unsafe_allow_html=True)

# =============================================================
# FOOTER
# =============================================================

st.markdown("""
<div class="footer">
    Machine Learning Car Price Prediction<br>
    <b>Disclaimer:</b>This price is an ML-based estimate and may differ from the actual market value. Use it as a reference only.
</div>
""", unsafe_allow_html=True)
