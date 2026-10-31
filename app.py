import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import sklearn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# ---------------------------------------------------------
# Page Configuration & Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="FraudGuard AI • Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional Blue-and-White Design System
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #0f172a;
    }
    
    /* Top Bar Gradient & Header */
    .app-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 50%, #1d4ed8 100%);
        padding: 28px 32px;
        border-radius: 16px;
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(30, 58, 138, 0.25);
    }
    .app-header h1 {
        color: #ffffff !important;
        font-weight: 800;
        font-size: 2.1rem;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .app-header p {
        color: #e0e7ff !important;
        font-size: 1.05rem;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Professional Card Containers */
    .white-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.04);
        margin-bottom: 20px;
    }
    .white-card:hover {
        border-color: #cbd5e1;
        box-shadow: 0 10px 15px -3px rgba(15, 23, 42, 0.07);
    }

    /* Metric Cards */
    .metric-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-top: 4px solid #2563eb;
        border-radius: 12px;
        padding: 18px 20px;
        transition: transform 0.15s ease;
    }
    .metric-box:hover {
        transform: translateY(-2px);
    }
    .metric-val {
        font-size: 1.85rem;
        font-weight: 700;
        color: #1e3a8a;
    }
    .metric-lbl {
        font-size: 0.88rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-top: 4px;
    }

    /* Power BI Canvas Styling */
    .pbi-card {
        background: #ffffff;
        border: 1px solid #d1d5db;
        border-top: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        text-align: center;
    }
    .pbi-num {
        font-size: 2rem;
        font-weight: 800;
        color: #111827;
    }
    .pbi-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
    }

    /* Status Badges */
    .badge-approved {
        background: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.88rem;
        display: inline-block;
    }
    .badge-review {
        background: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.88rem;
        display: inline-block;
    }
    .badge-declined {
        background: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.88rem;
        display: inline-block;
    }

    /* Result Action Banners */
    .banner-approved {
        background: #ecfdf5;
        border-left: 6px solid #10b981;
        padding: 20px;
        border-radius: 10px;
        margin-top: 15px;
    }
    .banner-review {
        background: #fffbeb;
        border-left: 6px solid #f59e0b;
        padding: 20px;
        border-radius: 10px;
        margin-top: 15px;
    }
    .banner-declined {
        background: #fef2f2;
        border-left: 6px solid #ef4444;
        padding: 20px;
        border-radius: 10px;
        margin-top: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Model Loading (Strictly original trained model via joblib)
# ---------------------------------------------------------
MODEL_FILE = "credit_card_fraud_detection_model.pkl"

@st.cache_resource(show_spinner="Loading trained machine learning pipeline...")
def load_trained_model():
    """
    Loads original trained model using joblib.load().
    Preserves exact trained pipeline.
    """
    if not os.path.exists(MODEL_FILE):
        return None, f"Model file '{MODEL_FILE}' not found in root directory."
    try:
        loaded = joblib.load(MODEL_FILE)
        return loaded, None
    except Exception as e:
        return None, str(e)


model, model_err = load_trained_model()

# Category definitions extracted from original trained pipeline encoder
MERCHANT_CATEGORIES = [
    'Auto & Transportation', 'Clothing & Apparel', 'Education',
    'Electronics & Technology', 'Entertainment & Streaming', 'Financial Services',
    'Gas Station', 'Grocery & Supermarket', 'Healthcare & Pharmacy',
    'Home Improvement', 'Hotel & Lodging', 'Online Retail',
    'Restaurant & Dining', 'Travel & Airlines', 'Utilities & Telecom'
]

MERCHANT_COUNTRIES = [
    'Australia', 'Brazil', 'Canada', 'China', 'France', 'Germany',
    'India', 'Italy', 'Japan', 'Mexico', 'Netherlands', 'Nigeria',
    'Romania', 'Russia', 'South Africa', 'South Korea', 'Spain',
    'UAE', 'UK', 'USA'
]

DAYS_OF_WEEK = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
GENDERS = ['Female', 'Male', 'Other']
STATES = [
    'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'FL', 'GA', 'HI', 'IA',
    'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'MI', 'MN',
    'MO', 'MS', 'NC', 'NE', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK',
    'OR', 'PA', 'SC', 'TN', 'TX', 'UT', 'VA', 'WA', 'WI', 'WV'
]
TRANSACTION_CHANNELS = ['ATM', 'In-store', 'Mobile App', 'Online']
DEVICE_TYPES = ['ATM', 'Desktop', 'Mobile', 'POS Terminal']
BROWSERS = ['App', 'Chrome', 'Edge', 'Firefox', 'Other', 'Safari']
EMAIL_DOMAINS = ['corporate', 'gmail', 'hotmail', 'other', 'yahoo']

EXPECTED_FEATURES = [
    'transaction_amount', 'merchant_category', 'merchant_country',
    'transaction_hour', 'transaction_day_of_week', 'card_present',
    'customer_age', 'customer_gender', 'customer_state', 'account_age_days',
    'credit_limit', 'credit_score', 'num_cards_held', 'transactions_last_1h',
    'transactions_last_24h', 'transactions_last_7d', 'amount_last_1h',
    'amount_last_24h', 'days_since_last_transaction', 'distance_from_home_km',
    'distance_from_last_tx_km', 'international_transaction',
    'high_risk_country', 'amount_vs_avg_ratio', 'used_chip', 'used_pin',
    'transaction_channel', 'device_type', 'browser', 'is_weekend',
    'avg_transaction_amount_30d', 'std_transaction_amount_30d',
    'num_declined_last_30d', 'num_chargebacks_ever', 'email_domain'
]


# ---------------------------------------------------------
# Sidebar Controls & Navigation
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=60)
    st.markdown("## **FraudGuard AI**")
    st.caption("Credit Card Fraud Prevention System")
    st.divider()

    menu = st.radio(
        "Application Navigation",
        [
            "🏠 Dashboard Home",
            "📊 Dataset & Fraud Analytics",
            "🔍 Single Transaction Prediction",
            "📁 Batch CSV Prediction",
            "📈 Model Performance & Metrics",
            "📊 Power BI Executive Dashboard",
            "ℹ️ Project & Methodology"
        ],
        index=0
    )

    st.divider()
    st.markdown("### ⚙️ **Decision Control**")
    threshold = st.slider(
        "Fraud Classification Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Probability threshold at or above which transactions are flagged as FRAUD."
    )
    st.caption(f"Active Threshold: **{threshold:.2f}** (Recommended: 0.50)")

    st.divider()
    st.markdown("### 🛡️ **System Telemetry**")
    if model is not None:
        st.success("✅ Model Pipeline Active")
        st.caption(f"• **Joblib:** `v{joblib.__version__}`")
        st.caption(f"• **Scikit-Learn:** `v{sklearn.__version__}`")
        st.caption("• **Architecture:** `Pipeline (ColumnTransformer + RandomForest)`")
        st.caption(f"• **Features Monitored:** `{len(EXPECTED_FEATURES)}`")
    else:
        st.error("❌ Model Load Failed")
        st.caption(str(model_err))


# ---------------------------------------------------------
# Header Banner
# ---------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>🛡️ FraudGuard AI • Enterprise Fraud Detection Platform</h1>
        <p>Real-time machine learning inference, transaction risk scoring, and behavioural anomaly detection</p>
    </div>
    """,
    unsafe_allow_html=True
)

if model is None:
    st.error(
        f"**Failed to load trained model pipeline:** `{model_err}`\n\n"
        f"Please verify that `{MODEL_FILE}` is placed in the project root folder."
    )
    st.stop()


# ---------------------------------------------------------
# SECTION 1: Dashboard Home
# ---------------------------------------------------------
if menu == "🏠 Dashboard Home":
    st.markdown("### 📊 Operational Overview")
    st.write(
        "Welcome to FraudGuard AI. This enterprise fraud prevention system evaluates credit card transactions "
        "in sub-milliseconds using an ensemble of 200 decision trees trained on multi-dimensional customer, "
        "geographic, velocity, and merchant risk attributes."
    )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(
            """
            <div class="metric-box">
                <div class="metric-val">35</div>
                <div class="metric-lbl">Monitored Attributes</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with kpi2:
        st.markdown(
            """
            <div class="metric-box">
                <div class="metric-val">200</div>
                <div class="metric-lbl">Random Forest Trees</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with kpi3:
        st.markdown(
            """
            <div class="metric-box">
                <div class="metric-val">&lt; 15 ms</div>
                <div class="metric-lbl">Inference Latency</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with kpi4:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-val">{threshold:.2f}</div>
                <div class="metric-lbl">Risk Cutoff Threshold</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🚀 Fast Navigation")

    card_c1, card_c2 = st.columns(2)
    with card_c1:
        st.markdown(
            """
            <div class="white-card">
                <h4 style="color: #1e3a8a; margin-top:0;">🔍 Single Transaction Evaluation</h4>
                <p style="color: #475569;">
                    Evaluate any live transaction in real-time. Test pre-configured legitimate, travel, 
                    and foreign velocity attack presets to view dynamic risk gauge and decision verdicts.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <div class="white-card">
                <h4 style="color: #1e3a8a; margin-top:0;">📊 Dataset & Fraud Analytics</h4>
                <p style="color: #475569;">
                    Inspect transaction distributions, high-risk merchant categories, geographic heatmaps, 
                    and hourly spend patterns using interactive Plotly charts.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with card_c2:
        st.markdown(
            """
            <div class="white-card">
                <h4 style="color: #1e3a8a; margin-top:0;">📁 Batch CSV Processing</h4>
                <p style="color: #475569;">
                    Score hundreds of transactions in seconds. Upload custom CSV files or load our 
                    pre-generated test dataset to inspect aggregate risk distributions and download scored results.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <div class="white-card">
                <h4 style="color: #1e3a8a; margin-top:0;">📊 Power BI Executive Dashboard</h4>
                <p style="color: #475569;">
                    View live embedded Power BI reports, interactive Power BI KPI canvases, 
                    DAX measure formulas, and ready-to-load datasets for Microsoft Power BI Desktop.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### 🛠️ Preprocessing & Classification Pipeline Architecture")
    p_step1, p_step2, p_step3 = st.columns(3)
    with p_step1:
        st.info(
            "**1. Numeric Preprocessing (26 Features)**\n\n"
            "• Missing values: `SimpleImputer(median)`\n"
            "• Scaling: `StandardScaler(z-score)`\n"
            "• Amounts, velocities, credit stats"
        )
    with p_step2:
        st.info(
            "**2. Categorical Preprocessing (9 Features)**\n\n"
            "• Missing values: `SimpleImputer(most_frequent)`\n"
            "• Encoding: `OneHotEncoder(handle_unknown='ignore')`\n"
            "• Expands to 104 binary indicators"
        )
    with p_step3:
        st.info(
            "**3. Random Forest Ensemble**\n\n"
            "• 200 Decision Trees (`max_depth=18`)\n"
            "• Balanced Subsample class weights\n"
            "• Probability calibration & voting"
        )


# ---------------------------------------------------------
# SECTION 2: Dataset & Fraud Analytics
# ---------------------------------------------------------
elif menu == "📊 Dataset & Fraud Analytics":
    st.markdown("### 📊 Dataset Overview & Transaction Analytics")
    st.write(
        "Interactive exploratory analytics showing transaction patterns, category risks, "
        "and velocity dynamics evaluated by the model."
    )

    data_path = "benchmark_validation_data.csv"
    if os.path.exists(data_path):
        df_analytics = pd.read_csv(data_path)
    else:
        df_analytics = pd.read_csv("sample_transactions.csv")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Sample Transactions", f"{len(df_analytics)}")
    s2.metric("Avg Transaction Amount", f"${df_analytics['transaction_amount'].mean():,.2f}")
    s3.metric("Max Transaction", f"${df_analytics['transaction_amount'].max():,.2f}")
    if 'is_fraud' in df_analytics.columns:
        fraud_pct = (df_analytics['is_fraud'].sum() / len(df_analytics)) * 100
        s4.metric("Fraud Incident Rate", f"{fraud_pct:.1f}%")
    else:
        s4.metric("Categories Monitored", f"{df_analytics['merchant_category'].nunique()}")

    st.markdown("---")

    chart_c1, chart_c2 = st.columns(2)
    with chart_c1:
        cat_group = df_analytics.groupby('merchant_category')['transaction_amount'].agg(['mean', 'count']).reset_index()
        cat_fig = px.bar(
            cat_group.sort_values(by='mean', ascending=False),
            x='merchant_category',
            y='mean',
            color='mean',
            color_continuous_scale=['#93c5fd', '#2563eb', '#1e3a8a'],
            labels={'mean': 'Average Amount ($)', 'merchant_category': 'Merchant Category'},
            title="Average Transaction Amount by Merchant Category"
        )
        cat_fig.update_layout(xaxis_tickangle=-45, font={'family': 'Plus Jakarta Sans'})
        st.plotly_chart(cat_fig, use_container_width=True)

    with chart_c2:
        scatter_fig = px.scatter(
            df_analytics,
            x='transaction_amount',
            y='distance_from_home_km',
            color='international_transaction',
            color_discrete_map={0: '#2563eb', 1: '#ef4444'},
            size='transactions_last_24h',
            hover_data=['merchant_country', 'customer_state'],
            labels={
                'transaction_amount': 'Transaction Amount ($)',
                'distance_from_home_km': 'Distance From Home (km)',
                'international_transaction': 'Is International'
            },
            title="Transaction Amount vs. Geographic Distance (Bubble: 24h Velocity)"
        )
        scatter_fig.update_layout(font={'family': 'Plus Jakarta Sans'})
        st.plotly_chart(scatter_fig, use_container_width=True)

    chart_c3, chart_c4 = st.columns(2)
    with chart_c3:
        hour_counts = df_analytics['transaction_hour'].value_counts().sort_index().reset_index()
        hour_counts.columns = ['Hour', 'Count']
        hour_fig = px.line(
            hour_counts,
            x='Hour',
            y='Count',
            markers=True,
            title="Transaction Activity by Hour of Day (0 - 23)",
            line_shape='spline'
        )
        hour_fig.update_traces(line_color='#2563eb', marker=dict(size=8, color='#1e3a8a'))
        hour_fig.update_layout(font={'family': 'Plus Jakarta Sans'})
        st.plotly_chart(hour_fig, use_container_width=True)

    with chart_c4:
        channel_counts = df_analytics['transaction_channel'].value_counts().reset_index()
        channel_counts.columns = ['Channel', 'Count']
        pie_fig = px.pie(
            channel_counts,
            values='Count',
            names='Channel',
            hole=0.45,
            color='Channel',
            color_discrete_sequence=['#1e3a8a', '#2563eb', '#60a5fa', '#93c5fd'],
            title="Transaction Volume by Channel Mix"
        )
        pie_fig.update_layout(font={'family': 'Plus Jakarta Sans'})
        st.plotly_chart(pie_fig, use_container_width=True)


# ---------------------------------------------------------
# SECTION 3: Single Transaction Fraud Prediction
# ---------------------------------------------------------
elif menu == "🔍 Single Transaction Prediction":
    st.markdown("### 🔍 Real-Time Single Transaction Scoring")
    st.write(
        "Enter individual transaction attributes or pick a pre-built profile to evaluate "
        "fraud likelihood with instant decision recommendations."
    )

    st.markdown("**⚡ Quick Load Profiles:**")
    p_col1, p_col2, p_col3 = st.columns(3)

    if "single_preset" not in st.session_state:
        st.session_state.single_preset = "normal"

    with p_col1:
        if st.button("🟢 Profile 1: Normal In-Store Grocery", use_container_width=True):
            st.session_state.single_preset = "normal"
    with p_col2:
        if st.button("🟡 Profile 2: Legitimate International Travel", use_container_width=True):
            st.session_state.single_preset = "travel"
    with p_col3:
        if st.button("🔴 Profile 3: Suspicious Foreign Velocity Spike", use_container_width=True):
            st.session_state.single_preset = "fraud"

    preset = st.session_state.single_preset
    if preset == "normal":
        def_amount, def_cat, def_country = 62.50, "Grocery & Supermarket", "USA"
        def_hour, def_day, def_card_present = 14, "Wednesday", True
        def_channel, def_device, def_browser = "In-store", "POS Terminal", "Other"
        def_chip, def_pin = True, True
        def_age, def_gender, def_state = 42, "Female", "CA"
        def_acc_days, def_limit, def_score, def_cards, def_domain = 1100, 15000.0, 750, 2, "gmail"
        def_dist_home, def_dist_last = 3.8, 2.1
        def_intl, def_high_risk = False, False
        def_tx_1h, def_tx_24h, def_tx_7d = 0, 2, 7
        def_amt_1h, def_amt_24h, def_days_last = 0.0, 85.0, 2
        def_avg_30d, def_std_30d = 58.0, 22.0
        def_declined, def_chargebacks = 0, 0

    elif preset == "travel":
        def_amount, def_cat, def_country = 380.00, "Hotel & Lodging", "France"
        def_hour, def_day, def_card_present = 19, "Friday", True
        def_channel, def_device, def_browser = "In-store", "POS Terminal", "Other"
        def_chip, def_pin = True, False
        def_age, def_gender, def_state = 53, "Male", "TX"
        def_acc_days, def_limit, def_score, def_cards, def_domain = 1950, 25000.0, 805, 3, "corporate"
        def_dist_home, def_dist_last = 6800.0, 95.0
        def_intl, def_high_risk = True, False
        def_tx_1h, def_tx_24h, def_tx_7d = 1, 3, 11
        def_amt_1h, def_amt_24h, def_days_last = 80.0, 520.0, 1
        def_avg_30d, def_std_30d = 160.0, 75.0
        def_declined, def_chargebacks = 0, 0

    else:  # fraud preset
        def_amount, def_cat, def_country = 3450.00, "Electronics & Technology", "Nigeria"
        def_hour, def_day, def_card_present = 3, "Sunday", False
        def_channel, def_device, def_browser = "Online", "Desktop", "Other"
        def_chip, def_pin = False, False
        def_age, def_gender, def_state = 23, "Male", "NY"
        def_acc_days, def_limit, def_score, def_cards, def_domain = 28, 4000.0, 550, 4, "other"
        def_dist_home, def_dist_last = 9200.0, 8100.0
        def_intl, def_high_risk = True, True
        def_tx_1h, def_tx_24h, def_tx_7d = 7, 16, 38
        def_amt_1h, def_amt_24h, def_days_last = 3100.0, 6200.0, 0
        def_avg_30d, def_std_30d = 65.0, 24.0
        def_declined, def_chargebacks = 5, 2

    with st.form("form_single_predict"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("#### 1️⃣ Transaction Core")
            in_amount = st.number_input("Transaction Amount ($)", min_value=0.5, max_value=100000.0, value=float(def_amount), step=10.0)
            in_category = st.selectbox("Merchant Category", MERCHANT_CATEGORIES, index=MERCHANT_CATEGORIES.index(def_cat))
            in_country = st.selectbox("Merchant Country", MERCHANT_COUNTRIES, index=MERCHANT_COUNTRIES.index(def_country))
            in_hour = st.slider("Hour of Transaction (0-23)", 0, 23, int(def_hour))
            in_day = st.selectbox("Day of Week", DAYS_OF_WEEK, index=DAYS_OF_WEEK.index(def_day))
            in_channel = st.selectbox("Transaction Channel", TRANSACTION_CHANNELS, index=TRANSACTION_CHANNELS.index(def_channel))
            in_device = st.selectbox("Device Type", DEVICE_TYPES, index=DEVICE_TYPES.index(def_device))
            in_browser = st.selectbox("Browser Interface", BROWSERS, index=BROWSERS.index(def_browser))
            
            sub1, sub2, sub3 = st.columns(3)
            with sub1:
                in_card_present = st.checkbox("Card Present", value=def_card_present)
            with sub2:
                in_chip = st.checkbox("Used Chip", value=def_chip)
            with sub3:
                in_pin = st.checkbox("Used PIN", value=def_pin)

        with c2:
            st.markdown("#### 2️⃣ Cardholder Profile")
            in_age = st.number_input("Customer Age", min_value=18, max_value=100, value=int(def_age))
            in_gender = st.selectbox("Customer Gender", GENDERS, index=GENDERS.index(def_gender))
            in_state = st.selectbox("Billing State (US)", STATES, index=STATES.index(def_state))
            in_domain = st.selectbox("Email Domain", EMAIL_DOMAINS, index=EMAIL_DOMAINS.index(def_domain))
            in_acc_days = st.number_input("Account Age (Days)", min_value=1, max_value=15000, value=int(def_acc_days))
            in_credit_limit = st.number_input("Credit Limit ($)", min_value=500.0, max_value=200000.0, value=float(def_limit), step=500.0)
            in_credit_score = st.slider("Credit Score (FICO)", 300, 850, int(def_score))
            in_cards_held = st.slider("Total Cards Held", 1, 10, int(def_cards))

        with c3:
            st.markdown("#### 3️⃣ Behavioral & Velocity Signals")
            in_dist_home = st.number_input("Distance from Home (km)", min_value=0.0, max_value=25000.0, value=float(def_dist_home), step=25.0)
            in_dist_last = st.number_input("Distance from Last Tx (km)", min_value=0.0, max_value=25000.0, value=float(def_dist_last), step=25.0)
            in_intl = st.checkbox("Cross-Border / International", value=def_intl)
            in_high_risk = st.checkbox("High-Risk Jurisdiction Flag", value=def_high_risk)
            
            vc1, vc2 = st.columns(2)
            with vc1:
                in_tx_1h = st.number_input("Tx Last 1h", min_value=0, max_value=50, value=int(def_tx_1h))
                in_tx_24h = st.number_input("Tx Last 24h", min_value=0, max_value=100, value=int(def_tx_24h))
                in_tx_7d = st.number_input("Tx Last 7d", min_value=0, max_value=500, value=int(def_tx_7d))
            with vc2:
                in_amt_1h = st.number_input("Amt Last 1h ($)", min_value=0.0, max_value=100000.0, value=float(def_amt_1h))
                in_amt_24h = st.number_input("Amt Last 24h ($)", min_value=0.0, max_value=100000.0, value=float(def_amt_24h))
                in_days_last = st.number_input("Days Since Last Tx", min_value=0, max_value=365, value=int(def_days_last))

            in_avg_30d = st.number_input("30-Day Avg Spend ($)", min_value=1.0, max_value=20000.0, value=float(def_avg_30d))
            in_std_30d = st.number_input("30-Day Spend Std Dev ($)", min_value=0.1, max_value=10000.0, value=float(def_std_30d))

            rc1, rc2 = st.columns(2)
            with rc1:
                in_declined = st.number_input("Declines (30d)", min_value=0, max_value=25, value=int(def_declined))
            with rc2:
                in_chargebacks = st.number_input("Prior Chargebacks", min_value=0, max_value=20, value=int(def_chargebacks))

        submit_predict = st.form_submit_button("⚡ Run Fraud Prediction", use_container_width=True, type="primary")

    if submit_predict:
        is_weekend_calc = 1 if in_day in ['Saturday', 'Sunday'] else 0
        ratio_calc = in_amount / in_avg_30d if in_avg_30d > 0 else 1.0

        single_input_dict = {
            'transaction_amount': [in_amount],
            'merchant_category': [in_category],
            'merchant_country': [in_country],
            'transaction_hour': [in_hour],
            'transaction_day_of_week': [in_day],
            'card_present': [1 if in_card_present else 0],
            'customer_age': [in_age],
            'customer_gender': [in_gender],
            'customer_state': [in_state],
            'account_age_days': [in_acc_days],
            'credit_limit': [in_credit_limit],
            'credit_score': [in_credit_score],
            'num_cards_held': [in_cards_held],
            'transactions_last_1h': [in_tx_1h],
            'transactions_last_24h': [in_tx_24h],
            'transactions_last_7d': [in_tx_7d],
            'amount_last_1h': [in_amt_1h],
            'amount_last_24h': [in_amt_24h],
            'days_since_last_transaction': [in_days_last],
            'distance_from_home_km': [in_dist_home],
            'distance_from_last_tx_km': [in_dist_last],
            'international_transaction': [1 if in_intl else 0],
            'high_risk_country': [1 if in_high_risk else 0],
            'amount_vs_avg_ratio': [ratio_calc],
            'used_chip': [1 if in_chip else 0],
            'used_pin': [1 if in_pin else 0],
            'transaction_channel': [in_channel],
            'device_type': [in_device],
            'browser': [in_browser],
            'is_weekend': [is_weekend_calc],
            'avg_transaction_amount_30d': [in_avg_30d],
            'std_transaction_amount_30d': [in_std_30d],
            'num_declined_last_30d': [in_declined],
            'num_chargebacks_ever': [in_chargebacks],
            'email_domain': [in_domain]
        }

        input_df = pd.DataFrame(single_input_dict)

        try:
            probabilities = model.predict_proba(input_df)[0]
            prob_fraud = float(probabilities[1])
            prob_legit = float(probabilities[0])
            is_fraud_decision = prob_fraud >= threshold

            st.markdown("---")
            st.markdown("### 🎯 Decision & Risk Classification")

            out_col1, out_col2 = st.columns([1, 1.25])

            with out_col1:
                prob_pct = prob_fraud * 100
                thresh_pct = threshold * 100

                if prob_pct < 25:
                    gauge_color = "#10b981"
                elif prob_pct < thresh_pct:
                    gauge_color = "#f59e0b"
                else:
                    gauge_color = "#ef4444"

                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=prob_pct,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Calculated Fraud Probability", 'font': {'size': 18, 'family': 'Plus Jakarta Sans'}},
                        number={'suffix': "%", 'font': {'size': 38, 'color': gauge_color, 'family': 'Plus Jakarta Sans'}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                            'bar': {'color': gauge_color, 'thickness': 0.35},
                            'bgcolor': "#ffffff",
                            'borderwidth': 1,
                            'bordercolor': "#cbd5e1",
                            'steps': [
                                {'range': [0, 25], 'color': "rgba(16, 185, 129, 0.12)"},
                                {'range': [25, thresh_pct], 'color': "rgba(245, 158, 11, 0.12)"},
                                {'range': [thresh_pct, 100], 'color': "rgba(239, 68, 68, 0.15)"},
                            ],
                            'threshold': {
                                'line': {'color': "#b91c1c", 'width': 3},
                                'thickness': 0.85,
                                'value': thresh_pct,
                            },
                        },
                    )
                )
                fig_gauge.update_layout(
                    height=250,
                    margin=dict(l=15, r=15, t=35, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={'family': 'Plus Jakarta Sans'},
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                st.write("**Model Confidence Breakdown:**")
                st.progress(prob_fraud)
                conf1, conf2 = st.columns(2)
                conf1.metric("Legitimate", f"{prob_legit*100:.2f}%")
                conf2.metric("Fraud", f"{prob_fraud*100:.2f}%")

            with out_col2:
                if is_fraud_decision:
                    st.markdown(
                        f"""
                        <div class="banner-declined">
                            <h2 style="color: #991b1b; margin: 0; font-weight:800;">🚫 TRANSACTION DECLINED</h2>
                            <p style="color: #7f1d1d; margin-top: 8px; font-size: 1.02rem;">
                                High probability of fraud detected (<strong>{prob_fraud*100:.2f}%</strong>), exceeding 
                                the active risk threshold of <strong>{threshold*100:.0f}%</strong>.
                            </p>
                            <span class="badge-declined">ACTION: BLOCK TRANSACTION</span>
                            <span class="badge-declined">DISPOSITION: ALERT FRAUD OPS</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                elif prob_fraud >= 0.25:
                    st.markdown(
                        f"""
                        <div class="banner-review">
                            <h2 style="color: #92400e; margin: 0; font-weight:800;">⚠️ STEP-UP AUTHENTICATION (2FA)</h2>
                            <p style="color: #78350f; margin-top: 8px; font-size: 1.02rem;">
                                Moderate risk detected (<strong>{prob_fraud*100:.2f}%</strong>). 
                                Suspicious signals observed. Hold transaction pending SMS OTP or biometric confirmation.
                            </p>
                            <span class="badge-review">ACTION: CHALLENGE CARDHOLDER</span>
                            <span class="badge-review">STATUS: PENDING 2FA</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="banner-approved">
                            <h2 style="color: #065f46; margin: 0; font-weight:800;">✅ TRANSACTION AUTHORIZED</h2>
                            <p style="color: #064e3b; margin-top: 8px; font-size: 1.02rem;">
                                Low fraud risk score (<strong>{prob_fraud*100:.2f}%</strong>). All velocity and geographic 
                                checks are within safe operating limits.
                            </p>
                            <span class="badge-approved">ACTION: APPROVE & SETTLE</span>
                            <span class="badge-approved">STATUS: NORMAL</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### 🔍 Notable Risk Drivers")
                risk_factors = []
                if in_dist_home > 3000:
                    risk_factors.append(f"🌐 Extreme distance from cardholder home: **{in_dist_home:,.0f} km**")
                if in_intl:
                    risk_factors.append(f"🌍 Foreign cross-border transaction ({in_country})")
                if in_high_risk:
                    risk_factors.append("🚨 High-risk jurisdiction list match")
                if ratio_calc > 3.0:
                    risk_factors.append(f"📈 Spend spike: **{ratio_calc:.1f}x** baseline 30-day average")
                if in_tx_1h >= 3:
                    risk_factors.append(f"⚡ High velocity: **{in_tx_1h} transactions** attempted in past hour")
                if not in_card_present:
                    risk_factors.append("💻 Card Not Present (CNP / e-commerce)")
                if in_declined > 0:
                    risk_factors.append(f"⚠️ **{in_declined}** recent card decline attempts")
                if in_chargebacks > 0:
                    risk_factors.append(f"📜 **{in_chargebacks}** historical chargeback on file")

                if risk_factors:
                    for rf in risk_factors:
                        st.markdown(f"- {rf}")
                else:
                    st.markdown("✅ No abnormal behavioural or velocity anomalies detected.")

        except Exception as pred_ex:
            st.error(f"Inference error: {pred_ex}")


# ---------------------------------------------------------
# SECTION 4: Batch CSV Prediction
# ---------------------------------------------------------
elif menu == "📁 Batch CSV Prediction":
    st.markdown("### 📁 Batch Transaction File Processing")
    st.write(
        "Upload a batch CSV file to score hundreds of transactions simultaneously. "
        "The model evaluates each row against all 35 features and computes individual fraud risk probabilities."
    )

    b_up_col, b_samp_col = st.columns([2, 1])

    with b_samp_col:
        st.markdown("#### 📥 Benchmark Data Templates")
        st.write("Need sample files to test?")
        try:
            with open("sample_transactions.csv", "rb") as f_samp:
                st.download_button(
                    "Download `sample_transactions.csv`",
                    data=f_samp,
                    file_name="sample_transactions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        except Exception:
            pass

        try:
            with open("benchmark_validation_data.csv", "rb") as f_bench:
                st.download_button(
                    "Download `benchmark_validation_data.csv`",
                    data=f_bench,
                    file_name="benchmark_validation_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        except Exception:
            pass

    with b_up_col:
        batch_file = st.file_uploader("Upload Transactions CSV", type=["csv"])

    df_batch = None
    if batch_file is not None:
        try:
            df_batch = pd.read_csv(batch_file)
            st.success(f"Uploaded `{batch_file.name}` ({len(df_batch)} transactions).")
        except Exception as e:
            st.error(f"Error parsing file: {e}")
    else:
        if st.button("📂 Or Load Benchmark Dataset Directly (60 Records)"):
            try:
                df_batch = pd.read_csv("benchmark_validation_data.csv")
                st.info(f"Loaded {len(df_batch)} benchmark transactions.")
            except Exception as e:
                st.error(f"Failed to load benchmark: {e}")

    if df_batch is not None:
        missing_features = [f for f in EXPECTED_FEATURES if f not in df_batch.columns]
        if missing_features:
            st.error(f"CSV is missing {len(missing_features)} required columns: `{missing_features}`")
        else:
            with st.spinner("Scoring batch transactions using original trained pipeline..."):
                x_batch = df_batch[EXPECTED_FEATURES].copy()
                probs = model.predict_proba(x_batch)[:, 1]
                preds = (probs >= threshold).astype(int)

                scored_df = df_batch.copy()
                scored_df["Fraud_Probability"] = np.round(probs, 4)
                scored_df["Risk_Score_%"] = np.round(probs * 100, 2)
                scored_df["Decision"] = np.where(preds == 1, "FRAUD / DECLINED", "APPROVED")

            total_tx = len(scored_df)
            fraud_tx = int(preds.sum())
            approved_tx = total_tx - fraud_tx
            fraud_rate = (fraud_tx / total_tx) * 100 if total_tx > 0 else 0.0
            prevented_loss = float(scored_df[preds == 1]["transaction_amount"].sum())

            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Transactions", f"{total_tx:,}")
            m2.metric("Approved", f"{approved_tx:,}", delta=f"{100 - fraud_rate:.1f}% approval")
            m3.metric("Flagged Fraud", f"{fraud_tx:,}", delta=f"-{fraud_rate:.1f}% risk", delta_color="inverse")
            m4.metric("Loss Prevented ($)", f"${prevented_loss:,.2f}")

            g_col1, g_col2 = st.columns(2)
            with g_col1:
                hist = px.histogram(
                    scored_df,
                    x="Risk_Score_%",
                    color="Decision",
                    color_discrete_map={"APPROVED": "#2563eb", "FRAUD / DECLINED": "#ef4444"},
                    nbins=20,
                    title="Risk Score Distribution Across Uploaded Batch"
                )
                hist.update_layout(bargap=0.1, font={'family': 'Plus Jakarta Sans'})
                st.plotly_chart(hist, use_container_width=True)

            with g_col2:
                donut = px.pie(
                    values=[approved_tx, fraud_tx],
                    names=["Approved", "Flagged Fraud"],
                    color=["Approved", "Flagged Fraud"],
                    color_discrete_map={"Approved": "#2563eb", "Flagged Fraud": "#ef4444"},
                    hole=0.5,
                    title="Transaction Disposition Proportion"
                )
                donut.update_layout(font={'family': 'Plus Jakarta Sans'})
                st.plotly_chart(donut, use_container_width=True)

            st.markdown("#### 📋 Scored Transactions Table")
            display_headers = [c for c in ["transaction_id", "transaction_amount", "merchant_category", "merchant_country", "Risk_Score_%", "Decision"] if c in scored_df.columns]
            st.dataframe(scored_df[display_headers], use_container_width=True)

            csv_out = scored_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Scored CSV with Probabilities",
                data=csv_out,
                file_name=f"scored_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="primary"
            )


# ---------------------------------------------------------
# SECTION 5: Model Performance & Metrics
# ---------------------------------------------------------
elif menu == "📈 Model Performance & Metrics":
    st.markdown("### 📈 Model Performance & Evaluation Metrics")
    st.write(
        "Model evaluation metrics are computed **live from actual model predictions** "
        "and true ground-truth labels using `scikit-learn.metrics`. No static or invented values are used."
    )

    eval_file = st.file_uploader("Upload Labeled Test Dataset (Optional)", type=["csv"], key="eval_uploader")

    df_eval = None
    label_col = None

    if eval_file is not None:
        try:
            df_eval = pd.read_csv(eval_file)
            st.success(f"Uploaded `{eval_file.name}` with {len(df_eval)} labeled records.")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    if df_eval is None:
        bench_path = "benchmark_validation_data.csv"
        if os.path.exists(bench_path):
            df_eval = pd.read_csv(bench_path)
            st.info(f"Loaded benchmark validation dataset ({len(df_eval)} transactions with ground-truth labels).")

    if df_eval is not None:
        potential_labels = ['is_fraud', 'fraud', 'Class', 'label', 'target', 'isFraud']
        for col in potential_labels:
            if col in df_eval.columns:
                label_col = col
                break

    if df_eval is None or label_col is None:
        st.warning("Please upload or load a labeled dataset containing ground-truth values ('is_fraud', 'Class', etc.).")
    else:
        x_eval = df_eval[EXPECTED_FEATURES]
        y_true = df_eval[label_col].values
        y_probs = model.predict_proba(x_eval)[:, 1]
        y_preds = (y_probs >= threshold).astype(int)

        acc = accuracy_score(y_true, y_preds)
        prec = precision_score(y_true, y_preds, zero_division=0)
        rec = recall_score(y_true, y_preds, zero_division=0)
        f1 = f1_score(y_true, y_preds, zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_probs)
        except Exception:
            auc = float('nan')

        st.markdown("#### 🎯 Computed Evaluation Metrics (Threshold = {:.2f})".format(threshold))
        pm1, pm2, pm3, pm4, pm5 = st.columns(5)
        pm1.metric("Accuracy", f"{acc*100:.2f}%")
        pm2.metric("Precision", f"{prec*100:.2f}%")
        pm3.metric("Recall", f"{rec*100:.2f}%")
        pm4.metric("F1-Score", f"{f1*100:.2f}%")
        pm5.metric("ROC-AUC", f"{auc:.4f}" if not np.isnan(auc) else "N/A")

        st.caption(
            "Metrics dynamically computed using `accuracy_score`, `precision_score`, `recall_score`, "
            "`f1_score`, and `roc_auc_score` from `sklearn.metrics` on ground-truth data."
        )

        st.markdown("---")

        c_eval1, c_eval2 = st.columns(2)

        with c_eval1:
            cm = confusion_matrix(y_true, y_preds)
            cm_fig = go.Figure(
                data=go.Heatmap(
                    z=cm,
                    x=['Predicted Legitimate (0)', 'Predicted Fraud (1)'],
                    y=['Actual Legitimate (0)', 'Actual Fraud (1)'],
                    colorscale=[[0, "#eff6ff"], [0.5, "#60a5fa"], [1, "#1e3a8a"]],
                    text=[[f"{cm[0][0]}<br>(TN)", f"{cm[0][1]}<br>(FP)"],
                          [f"{cm[1][0]}<br>(FN)", f"{cm[1][1]}<br>(TP)"]],
                    texttemplate="%{text}",
                    textfont={"size": 18, "family": "Plus Jakarta Sans"},
                    showscale=False
                )
            )
            cm_fig.update_layout(
                title="Confusion Matrix Heatmap",
                xaxis_title="Predicted Label",
                yaxis_title="True Label",
                font={'family': 'Plus Jakarta Sans'},
                height=380
            )
            st.plotly_chart(cm_fig, use_container_width=True)

        with c_eval2:
            fpr, tpr, _ = roc_curve(y_true, y_probs)
            roc_fig = go.Figure()
            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Random Forest (AUC = {auc:.3f})', line=dict(color='#2563eb', width=3)))
            roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Chance (AUC = 0.500)', line=dict(dash='dash', color='#94a3b8')))
            roc_fig.update_layout(
                title="Receiver Operating Characteristic (ROC) Curve",
                xaxis_title="False Positive Rate (1 - Specificity)",
                yaxis_title="True Positive Rate (Recall)",
                font={'family': 'Plus Jakarta Sans'},
                height=380
            )
            st.plotly_chart(roc_fig, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🏆 Model Feature Importances")
        try:
            rf_clf = model.named_steps['classifier']
            preproc = model.named_steps['preprocessor']
            feat_names = preproc.get_feature_names_out()
            importances = pd.Series(rf_clf.feature_importances_, index=feat_names).sort_values(ascending=False).head(15)

            imp_df = importances.reset_index()
            imp_df.columns = ["Feature", "Importance"]
            imp_df["Feature"] = imp_df["Feature"].str.replace("num__", "").str.replace("cat__", "")

            imp_fig = px.bar(
                imp_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale=['#bfdbfe', '#2563eb', '#1e3a8a'],
                title="Top 15 Predictive Features Extracted from Trained Decision Trees"
            )
            imp_fig.update_layout(yaxis={'categoryorder': 'total ascending'}, font={'family': 'Plus Jakarta Sans'}, height=450)
            st.plotly_chart(imp_fig, use_container_width=True)
        except Exception as ex_fi:
            st.caption(f"Feature importance display: {ex_fi}")


# ---------------------------------------------------------
# SECTION 6: Power BI Executive Dashboard
# ---------------------------------------------------------
elif menu == "📊 Power BI Executive Dashboard":
    st.markdown("### 📊 Power BI Executive Business Intelligence")
    st.write(
        "Interactive Power BI integration hub: embed live Power BI Service reports, "
        "explore an interactive Power BI canvas with DAX metrics, or export clean datasets with ready-to-use DAX measures."
    )

    pbi_tab1, pbi_tab2, pbi_tab3 = st.tabs([
        "🖥️ Live Power BI Embed",
        "📈 Power BI Visual Analytics Canvas",
        "📐 DAX Formulas & Power BI Desktop Toolkit"
    ])

    # --- TAB 1: Live Power BI Embed ---
    with pbi_tab1:
        st.markdown("#### 🌐 Embedded Power BI Report Viewer")
        st.write("Embed any interactive Power BI report published to the web or hosted on Power BI Service.")

        pbi_url_input = st.text_input(
            "Power BI Report URL (Publish to Web or Service Embed Link):",
            value="",
            placeholder="https://app.powerbi.com/view?r=eyJrIjoi...",
            help="Paste your Power BI Publish to Web URL or report embed URL here."
        )

        col_h1, col_h2 = st.columns([3, 1])
        with col_h2:
            frame_height = st.slider("Report Canvas Height (px)", min_value=450, max_value=1000, value=650, step=50)

        if pbi_url_input.strip():
            clean_url = pbi_url_input.strip()
            # If the user pasted an iframe tag, extract the src URL
            if "<iframe" in clean_url and "src=" in clean_url:
                import re
                src_match = re.search(r'src="([^"]+)"', clean_url)
                if src_match:
                    clean_url = src_match.group(1)

            st.success(f"Rendering Power BI Report from: `{clean_url[:65]}...`")
            components.html(
                f"""
                <div style="display:flex; justify-content:center; align-items:center; width:100%; height:{frame_height}px; border-radius:12px; overflow:hidden; box-shadow:0 8px 16px rgba(0,0,0,0.1); border:1px solid #cbd5e1;">
                    <iframe title="Power BI Fraud Analytics Report" width="100%" height="{frame_height}px" src="{clean_url}" frameborder="0" allowFullScreen="true"></iframe>
                </div>
                """,
                height=frame_height + 20,
            )
        else:
            st.info(
                "💡 **No external Power BI URL entered.** Enter your Power BI Publish-to-Web link above to load your live report directly. "
                "In the meantime, view the pre-built **Power BI Visual Analytics Canvas** in the next tab!"
            )
            # Show high-fidelity preview frame
            components.html(
                f"""
                <div style="background: linear-gradient(135deg, #1f2937 0%, #111827 100%); width:100%; height:480px; border-radius:12px; display:flex; flex-direction:column; justify-content:center; align-items:center; color:#ffffff; font-family:'Segoe UI', sans-serif; text-align:center; padding:30px; box-shadow:0 8px 24px rgba(0,0,0,0.15); border:1px solid #374151;">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/c/cf/New_Power_BI_Logo.svg" width="70" style="margin-bottom:15px;"/>
                    <h2 style="font-weight:700; margin:0 0 10px 0; color:#f3f4f6;">Microsoft Power BI Embedded Workspace</h2>
                    <p style="color:#9ca3af; max-width:600px; font-size:14px; line-height:1.6;">
                        Connect your live workspace report by pasting your <strong>Publish to Web</strong> embed link into the input box above. 
                        Live slicers, drill-through cards, and DAX KPIs will render directly inside FraudGuard AI.
                    </p>
                    <div style="margin-top:20px; display:flex; gap:12px;">
                        <span style="background:rgba(245, 158, 11, 0.2); color:#fbbf24; border:1px solid #f59e0b; padding:6px 14px; border-radius:6px; font-size:13px; font-weight:600;">Power BI Service</span>
                        <span style="background:rgba(59, 130, 246, 0.2); color:#60a5fa; border:1px solid #3b82f6; padding:6px 14px; border-radius:6px; font-size:13px; font-weight:600;">DirectQuery & Import</span>
                        <span style="background:rgba(16, 185, 129, 0.2); color:#34d399; border:1px solid #10b981; padding:6px 14px; border-radius:6px; font-size:13px; font-weight:600;">Row-Level Security</span>
                    </div>
                </div>
                """,
                height=500
            )

    # --- TAB 2: Power BI Visual Analytics Canvas ---
    with pbi_tab2:
        st.markdown("#### 📈 Enterprise Power BI Executive Board")
        st.write("Native Power BI styled dashboard with cross-filtering slicers, DAX KPI measure tiles, and drilldowns.")

        # Load Power BI dataset
        pbi_file = "powerbi_fraud_dataset.csv"
        if os.path.exists(pbi_file):
            df_pbi = pd.read_csv(pbi_file)
        else:
            df_pbi = pd.read_csv("benchmark_validation_data.csv")
            probs = model.predict_proba(df_pbi[EXPECTED_FEATURES])[:, 1]
            df_pbi['Risk_Score_%'] = np.round(probs * 100, 2)
            df_pbi['Decision'] = np.where(probs >= threshold, 'FRAUD / DECLINED', 'APPROVED')
            df_pbi['Loss_Prevented'] = np.where(probs >= threshold, df_pbi['transaction_amount'], 0.0)

        # Power BI Slicers Header
        st.markdown("##### 🎛️ **Power BI Slicers (Interactive Filters)**")
        sl_c1, sl_c2, sl_c3, sl_c4 = st.columns(4)

        with sl_c1:
            all_cats = ["(All Categories)"] + sorted(df_pbi['merchant_category'].unique().tolist())
            sel_cat = st.selectbox("Merchant Category", all_cats, index=0)
        with sl_c2:
            all_channels = ["(All Channels)"] + sorted(df_pbi['transaction_channel'].unique().tolist())
            sel_chan = st.selectbox("Channel", all_channels, index=0)
        with sl_c3:
            all_countries = ["(All Countries)"] + sorted(df_pbi['merchant_country'].unique().tolist())
            sel_ctry = st.selectbox("Country", all_countries, index=0)
        with sl_c4:
            all_decisions = ["(All Decisions)"] + sorted(df_pbi['Decision'].unique().tolist())
            sel_dec = st.selectbox("Decision Status", all_decisions, index=0)

        # Apply Slicers
        df_filtered = df_pbi.copy()
        if sel_cat != "(All Categories)":
            df_filtered = df_filtered[df_filtered['merchant_category'] == sel_cat]
        if sel_chan != "(All Channels)":
            df_filtered = df_filtered[df_filtered['transaction_channel'] == sel_chan]
        if sel_ctry != "(All Countries)":
            df_filtered = df_filtered[df_filtered['merchant_country'] == sel_ctry]
        if sel_dec != "(All Decisions)":
            df_filtered = df_filtered[df_filtered['Decision'] == sel_dec]

        st.caption(f"Showing **{len(df_filtered)}** of **{len(df_pbi)}** transactions after slicer filtering.")

        # Power BI KPI Cards
        st.markdown("---")
        k_tot_vol = df_filtered['transaction_amount'].sum()
        k_loss_prev = df_filtered[df_filtered['Decision'] == 'FRAUD / DECLINED']['transaction_amount'].sum()
        k_fraud_cnt = (df_filtered['Decision'] == 'FRAUD / DECLINED').sum()
        k_rate = (k_fraud_cnt / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
        k_avg_score = df_filtered['Risk_Score_%'].mean() if len(df_filtered) > 0 else 0

        pbi_k1, pbi_k2, pbi_k3, pbi_k4, pbi_k5 = st.columns(5)
        with pbi_k1:
            st.markdown(
                f"""
                <div class="pbi-card">
                    <div class="pbi-num">${k_tot_vol:,.0f}</div>
                    <div class="pbi-label">Total Volume Transacted</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with pbi_k2:
            st.markdown(
                f"""
                <div class="pbi-card" style="border-top-color: #ef4444;">
                    <div class="pbi-num" style="color:#dc2626;">${k_loss_prev:,.0f}</div>
                    <div class="pbi-label">Fraud Loss Prevented</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with pbi_k3:
            st.markdown(
                f"""
                <div class="pbi-card" style="border-top-color: #2563eb;">
                    <div class="pbi-num" style="color:#1d4ed8;">{k_fraud_cnt}</div>
                    <div class="pbi-label">Flagged Fraud Attempts</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with pbi_k4:
            st.markdown(
                f"""
                <div class="pbi-card" style="border-top-color: #10b981;">
                    <div class="pbi-num" style="color:#047857;">{k_rate:.1f}%</div>
                    <div class="pbi-label">Fraud Incident Rate</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with pbi_k5:
            st.markdown(
                f"""
                <div class="pbi-card" style="border-top-color: #8b5cf6;">
                    <div class="pbi-num" style="color:#6d28d9;">{k_avg_score:.1f}%</div>
                    <div class="pbi-label">Avg Anomaly Score</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Power BI Multi-Visual Grid
        pv_c1, pv_c2 = st.columns(2)
        with pv_c1:
            # Power BI Treemap of Fraud Loss by Category
            fraud_cat = df_filtered[df_filtered['Decision'] == 'FRAUD / DECLINED'].groupby('merchant_category')['transaction_amount'].sum().reset_index()
            if not fraud_cat.empty:
                tree_fig = px.treemap(
                    fraud_cat,
                    path=['merchant_category'],
                    values='transaction_amount',
                    color='transaction_amount',
                    color_continuous_scale=['#fef3c7', '#f59e0b', '#dc2626'],
                    title="Power BI Treemap: Fraud Loss Exposure by Merchant Category ($)"
                )
                tree_fig.update_layout(font={'family': 'Segoe UI'})
                st.plotly_chart(tree_fig, use_container_width=True)
            else:
                st.info("No flagged fraud transactions matching current slicer criteria.")

        with pv_c2:
            # Power BI Stacked Bar of Channel vs Decision
            chan_dec = df_filtered.groupby(['transaction_channel', 'Decision']).size().reset_index(name='Count')
            bar_fig = px.bar(
                chan_dec,
                x='transaction_channel',
                y='Count',
                color='Decision',
                color_discrete_map={"APPROVED": "#2563eb", "FRAUD / DECLINED": "#ef4444"},
                barmode='group',
                title="Transaction Channel Risk Matrix (Approved vs. Declined)"
            )
            bar_fig.update_layout(font={'family': 'Segoe UI'})
            st.plotly_chart(bar_fig, use_container_width=True)

        pv_c3, pv_c4 = st.columns(2)
        with pv_c3:
            # Country Risk Heatmap
            country_risk = df_filtered.groupby('merchant_country')['Risk_Score_%'].mean().reset_index()
            ctry_fig = px.bar(
                country_risk.sort_values(by='Risk_Score_%', ascending=False).head(10),
                x='merchant_country',
                y='Risk_Score_%',
                color='Risk_Score_%',
                color_continuous_scale=['#bfdbfe', '#2563eb', '#1e3a8a'],
                title="Top 10 Geographic Jurisdictions by Average Fraud Risk Score (%)"
            )
            ctry_fig.update_layout(font={'family': 'Segoe UI'}, xaxis_tickangle=-45)
            st.plotly_chart(ctry_fig, use_container_width=True)

        with pv_c4:
            # Card Present vs Distance Bubble Chart
            dist_fig = px.scatter(
                df_filtered,
                x='distance_from_home_km',
                y='amount_vs_avg_ratio',
                color='Decision',
                color_discrete_map={"APPROVED": "#10b981", "FRAUD / DECLINED": "#ef4444"},
                size='transaction_amount',
                hover_data=['merchant_category', 'customer_state'],
                labels={'distance_from_home_km': 'Distance from Home (km)', 'amount_vs_avg_ratio': 'Spend Spike Ratio'},
                title="Spend Anomaly vs Distance from Home (Bubble Size: Amount)"
            )
            dist_fig.update_layout(font={'family': 'Segoe UI'})
            st.plotly_chart(dist_fig, use_container_width=True)

    # --- TAB 3: DAX Formulas & Power BI Desktop Toolkit ---
    with pbi_tab3:
        st.markdown("#### 📐 Power BI Desktop Integration & DAX Measures")
        st.write(
            "Use this toolkit to connect Microsoft Power BI Desktop directly to your fraud detection dataset "
            "and create enterprise reports with pre-built DAX formulas."
        )

        dax_col1, dax_col2 = st.columns([1.2, 1])

        with dax_col2:
            st.markdown("##### 📥 Power BI Formatted Dataset")
            st.write("Download the enriched dataset containing calculated fields for Power BI modeling:")
            try:
                with open("powerbi_fraud_dataset.csv", "rb") as f_pbi_data:
                    st.download_button(
                        "Download `powerbi_fraud_dataset.csv`",
                        data=f_pbi_data,
                        file_name="powerbi_fraud_dataset.csv",
                        mime="text/csv",
                        type="primary",
                        use_container_width=True
                    )
            except Exception:
                st.caption("powerbi_fraud_dataset.csv generating...")

            st.markdown("##### 🚀 3-Step Power BI Desktop Setup")
            st.markdown(
                """
                1. **Import Data:** Open Power BI Desktop $\\rightarrow$ *Get Data* $\\rightarrow$ *Text/CSV* $\\rightarrow$ select `powerbi_fraud_dataset.csv`.
                2. **Create Measures:** Click *New Measure* on the Modeling tab and paste the DAX formulas on the left.
                3. **Publish to Service:** Click *Publish* $\\rightarrow$ Select workspace $\\rightarrow$ Copy the embed link into **Tab 1**!
                """
            )

        with dax_col1:
            st.markdown("##### 📋 Ready-to-Copy DAX Measures")
            st.code(
                """
// 1. Total Transaction Count
Total Transactions = COUNTROWS('Transactions')

// 2. Total Fraud Attempts Blocked
Fraud Transactions = 
CALCULATE(
    COUNTROWS('Transactions'), 
    'Transactions'[Decision] = "FRAUD / DECLINED"
)

// 3. Overall Fraud Rate (%)
Fraud Rate % = 
DIVIDE(
    [Fraud Transactions], 
    [Total Transactions], 
    0
) * 100

// 4. Financial Fraud Loss Prevented ($)
Loss Prevented ($) = 
CALCULATE(
    SUM('Transactions'[transaction_amount]), 
    'Transactions'[Decision] = "FRAUD / DECLINED"
)

// 5. Total Volume Processed ($)
Total Volume ($) = SUM('Transactions'[transaction_amount])

// 6. Average Fraud Risk Index
Average Risk Index = AVERAGE('Transactions'[Risk_Score_%])

// 7. High Risk Exposure Ratio
High Risk Volume % = 
DIVIDE(
    CALCULATE(SUM('Transactions'[transaction_amount]), 'Transactions'[Risk_Score_%] >= 50),
    [Total Volume ($)],
    0
) * 100
                """,
                language="dax"
            )


# ---------------------------------------------------------
# SECTION 7: Project Information & Methodology
# ---------------------------------------------------------
elif menu == "ℹ️ Project & Methodology":
    st.markdown("### ℹ️ Project Information & Methodology")
    
    st.markdown("#### 🎯 Business Context & Problem Statement")
    st.write(
        "Credit card fraud results in tens of billions of dollars in losses annually. Traditional rule-based systems "
        "suffer from rigidity and high false-positive rates that inconvenience legitimate cardholders. "
        "This project employs a machine learning approach that balances sensitivity to anomalous activity "
        "with precision on standard spending patterns."
    )

    st.markdown("#### ⚙️ Data Engineering & Feature Preprocessing")
    st.write(
        "The model uses a Scikit-Learn `ColumnTransformer` that executes targeted pipelines for both numeric and categorical variables:"
    )
    
    col_meth1, col_meth2 = st.columns(2)
    with col_meth1:
        st.markdown(
            """
            <div class="white-card">
                <h4 style="color:#1e3a8a;">Numeric Pipeline (26 Features)</h4>
                <ul>
                    <li><strong>SimpleImputer(strategy='median'):</strong> Robust against heavy-tailed financial distributions.</li>
                    <li><strong>StandardScaler():</strong> Zero-mean and unit-variance scaling for velocity counters and dollar amounts.</li>
                    <li><strong>Features:</strong> Transaction amounts, 1h/24h/7d counts, distances from home and prior tx, credit limits, account age, credit scores.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_meth2:
        st.markdown(
            """
            <div class="white-card">
                <h4 style="color:#1e3a8a;">Categorical Pipeline (9 Features)</h4>
                <ul>
                    <li><strong>SimpleImputer(strategy='most_frequent'):</strong> Handles missing categorical tokens.</li>
                    <li><strong>OneHotEncoder(handle_unknown='ignore'):</strong> Expands multi-class merchant categories, US states, channels, and device types into binary flags.</li>
                    <li><strong>Features:</strong> Merchant category, merchant country, day of week, channel, device, browser, gender, state, email domain.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("#### 🌲 Classification Algorithm")
    st.write(
        "A **RandomForestClassifier** with 200 estimators was trained using `balanced_subsample` weighting to handle severe class imbalance. "
        "Subsampling ensures each decision tree sees a balanced subset of legitimate and fraudulent transactions."
    )

    st.markdown("#### ☁️ Streamlit Community Cloud Deployment Guide")
    st.code(
        """
# To deploy on Streamlit Community Cloud:
1. Push this repository to GitHub containing:
   - app.py
   - requirements.txt
   - credit_card_fraud_detection_model.pkl
   - sample_transactions.csv
   - benchmark_validation_data.csv
   - powerbi_fraud_dataset.csv
   - README.md
2. Log into https://share.streamlit.io/
3. Select your repository, branch 'main', and entry file 'app.py'
4. Click 'Deploy'
        """,
        language="bash"
    )

# Footer
st.markdown("---")
st.caption(
    "FraudGuard AI • Enterprise Credit Card Fraud Detection Platform • Built with Streamlit, Scikit-learn, Plotly & Power BI Integration"
)
