# 🛡️ FraudGuard AI • Credit Card Fraud Detection Web Application

FraudGuard AI is an enterprise-grade credit card fraud detection web application powered by **Streamlit**, **Scikit-Learn**, **Plotly**, and **Microsoft Power BI**. The application uses an actual trained **RandomForestClassifier** pipeline inside a **ColumnTransformer** to evaluate transactions across 35 behavioral, geographic, temporal, and velocity features in real-time.

---

## 🌟 Application Features

1. **Professional Home Dashboard**:
   - Executive telemetry metrics: 35 monitored signals, 200 ensemble trees, sub-15ms inference latency, customizable decision threshold.
   - Quick navigation cards to every section of the platform.
   - Pipeline architecture flow chart and data processing overview.

2. **Dataset Overview & Fraud Analytics**:
   - Transaction volume and spending metrics (average, maximum, incident rate).
   - Interactive Plotly visualizations:
     - Average transaction amount by merchant category.
     - Transaction amount vs. geographic distance from home with velocity bubbles.
     - 24-hour transaction activity patterns.
     - Transaction distribution across channels (In-store, Online, Mobile App, ATM).

3. **Single Transaction Fraud Prediction**:
   - Comprehensive multi-card input interface covering all 35 features:
     - **Transaction Core:** Amount, category, country, hour, day, channel, device, browser, chip/PIN, card present.
     - **Cardholder Profile:** Age, gender, state, email domain, account age, credit limit, credit score, cards held.
     - **Behavioral & Velocity Signals:** Distances (home & last tx), cross-border flags, 1h/24h/7d counts and dollar volumes, 30-day averages, declines, chargebacks.
   - 3 Quick-Load Scenario Presets:
     - 🟢 *Normal In-Store Grocery*
     - 🟡 *Legitimate International Travel*
     - 🔴 *Suspicious Foreign Velocity Spike*
   - Interactive Plotly Fraud Risk Score Gauge (0% – 100%).
   - Actionable disposition banners (Authorized, Step-Up 2FA, Blocked/Declined).
   - Notable risk driver indicators detailing contributing anomaly signals.

4. **Batch CSV Prediction**:
   - Score hundreds of transactions concurrently via CSV file upload.
   - Includes one-click test loading and download for `sample_transactions.csv` and `benchmark_validation_data.csv`.
   - Summary KPIs: Total transactions, approved count, flagged fraud count, estimated fraud losses prevented ($).
   - Risk score histogram and approval ratio donut chart.
   - Filterable results table with individual fraud probabilities.
   - Instant export of scored transactions with timestamps.

5. **Model Performance & Evaluation Metrics**:
   - Computes **Accuracy, Precision, Recall, F1-Score, and ROC-AUC** live using `sklearn.metrics`.
   - Strictly avoids static or invented performance metrics.
   - Interactive **Confusion Matrix Heatmap** (TN, FP, FN, TP).
   - Interactive **Receiver Operating Characteristic (ROC) Curve** with AUC score.
   - **Feature Importance Chart** displaying the top 15 most influential features extracted directly from the trained decision trees.
   - Dynamic threshold slider allows real-time metric recalculation.

6. **📊 Power BI Executive Dashboard**:
   - **🖥️ Live Power BI Embed:** Seamlessly embed any live Power BI Service report or Publish-to-Web URL directly within the web app with custom height and full-screen controls.
   - **📈 Power BI Visual Analytics Canvas:** Native Power BI-style executive board with interactive cross-filtering Slicers (Category, Channel, Country, Decision Status), DAX KPI measure tiles, loss treemaps, channel risk matrices, and country risk heatmaps.
   - **📐 DAX Formulas & Power BI Desktop Toolkit:** Ready-to-copy DAX measures (Total Volume, Loss Prevented, Fraud Rate %, High Risk Volume %) and a 1-click download of `powerbi_fraud_dataset.csv` pre-formatted for Power BI Desktop modeling.

7. **Project Information & Methodology**:
   - In-depth documentation of the problem statement and business value.
   - Detailed specifications for the numeric pipeline (median imputer + standard scaler) and categorical pipeline (most-frequent imputer + one-hot encoder).
   - Model hyperparameters (200 estimators, max depth 18, balanced subsample weights).
   - Streamlit Community Cloud deployment instructions.

---

## 🎨 Design System

The application uses a **professional blue-and-white enterprise theme**:
- **Primary Brand Accents:** Deep Royal Navy (`#1e3a8a`), Sapphire Blue (`#2563eb`), and Soft Cobalt (`#3b82f6`).
- **Canvas & Containers:** Pure White (`#ffffff`) card surfaces with Slate (`#f8fafc`) page backdrops and subtle borders (`#e2e8f0`).
- **Typography:** Plus Jakarta Sans for sharp, modern enterprise readability.
- **Visuals:** High-contrast, interactive Plotly charts tailored to the blue-and-white aesthetic.

---

## 🔬 Model Specification

- **File:** `credit_card_fraud_detection_model.pkl`
- **Serialization:** `joblib`
- **Pipeline Type:** `sklearn.pipeline.Pipeline`
- **Preprocessor:** `sklearn.compose.ColumnTransformer`
  - **Numeric Features (26):** `transaction_amount`, `transaction_hour`, `card_present`, `customer_age`, `account_age_days`, `credit_limit`, `credit_score`, `num_cards_held`, `transactions_last_1h`, `transactions_last_24h`, `transactions_last_7d`, `amount_last_1h`, `amount_last_24h`, `days_since_last_transaction`, `distance_from_home_km`, `distance_from_last_tx_km`, `international_transaction`, `high_risk_country`, `amount_vs_avg_ratio`, `used_chip`, `used_pin`, `is_weekend`, `avg_transaction_amount_30d`, `std_transaction_amount_30d`, `num_declined_last_30d`, `num_chargebacks_ever`.
  - **Categorical Features (9):** `merchant_category`, `merchant_country`, `transaction_day_of_week`, `customer_gender`, `customer_state`, `transaction_channel`, `device_type`, `browser`, `email_domain`.
- **Classifier:** `sklearn.ensemble.RandomForestClassifier(n_estimators=200, max_depth=18, min_samples_leaf=5, class_weight='balanced_subsample', random_state=42)`

---

## 🚀 Getting Started

### 1. Local Setup

Clone or open the repository folder:
```powershell
# Activate your virtual environment
.\.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt

# Launch the Streamlit application
streamlit run app.py
```
Or simply double-click `run_app.bat`.

The application will be accessible at `http://localhost:8501`.

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push this directory to a GitHub repository containing:
   - `app.py`
   - `requirements.txt`
   - `credit_card_fraud_detection_model.pkl`
   - `sample_transactions.csv`
   - `benchmark_validation_data.csv`
   - `powerbi_fraud_dataset.csv`
   - `README.md`
2. Navigate to [share.streamlit.io](https://share.streamlit.io/) and connect your GitHub account.
3. Select your repository, set the branch to `main`, and specify `app.py` as the main file path.
4. Click **Deploy**.
