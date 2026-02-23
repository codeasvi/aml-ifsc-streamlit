import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------------------------------
# CONFIG
# ------------------------------------------------
st.set_page_config(page_title="IFSC AML Monitoring (Advanced)", layout="wide")
st.title("IFSC Risk-Based AML Monitoring System — Advanced")

# ------------------------------------------------
# FAST UPLOAD + VALIDATION
# ------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV (creditcard.csv)", type=["csv"])

if uploaded_file is None:
    st.info("Upload dataset to start.")
    st.stop()

@st.cache_data
def load_df(file):
    df = pd.read_csv(file)
    return df

df = load_df(uploaded_file)

required_cols = ["Time", "Amount"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

if "Class" not in df.columns:
    df["Class"] = 0  # fallback if labels absent

# ------------------------------------------------
# AML CONTEXT SIMULATION (GEOGRAPHY RISK)
# ------------------------------------------------
countries = ["India","UAE","UK","Singapore","Cayman Islands","Panama"]
high_risk = ["Cayman Islands","Panama"]

np.random.seed(42)
df["origin_country"] = np.random.choice(countries, len(df))
df["destination_country"] = np.random.choice(countries, len(df))
df["high_risk_country_flag"] = df["destination_country"].isin(high_risk).astype(int)

# ------------------------------------------------
# DYNAMIC THRESHOLDS (NOT STATIC)
# ------------------------------------------------
percentile = st.sidebar.slider("Investigation Threshold Percentile", 80, 99, 95)
investigate_threshold = df["Amount"].quantile(percentile/100)
high_threshold = df["Amount"].quantile(0.99)

# ------------------------------------------------
# BEHAVIORAL BASELINE MODELING
# ------------------------------------------------
global_avg = df["Amount"].mean()
df["behavior_deviation"] = df["Amount"] / (global_avg + 1e-6)

# ------------------------------------------------
# VELOCITY (SMURFING) DETECTION
# ------------------------------------------------
df = df.sort_values("Time")
df["time_diff"] = df["Time"].diff().fillna(99999)
df["velocity_flag"] = (df["time_diff"] < 10).astype(int)

# ------------------------------------------------
# AMOUNT RISK CLASSIFICATION
# ------------------------------------------------
def amount_risk(amount):
    if amount > high_threshold:
        return "High Risk – Immediate Review"
    elif amount > investigate_threshold:
        return "Higher Risk – Investigate"
    else:
        return "Normal"

df["amount_risk_flag"] = df["Amount"].apply(amount_risk)

# ------------------------------------------------
# ML ANOMALY DETECTION
# ------------------------------------------------
model = IsolationForest(contamination=0.01, random_state=42)
df["anomaly"] = model.fit_predict(df[["Amount","Time"]])

# ------------------------------------------------
# HYBRID RISK SCORE (RBA MODEL)
# ------------------------------------------------
def risk_engine(row):
    score = 0
    reasons = []

    if row["amount_risk_flag"] != "Normal":
        score += 25
        reasons.append("High Amount")

    if row["behavior_deviation"] > 5:
        score += 20
        reasons.append("Behavior Deviation")

    if row["velocity_flag"] == 1:
        score += 15
        reasons.append("High Velocity")

    if row["high_risk_country_flag"] == 1:
        score += 20
        reasons.append("High-Risk Geography")

    if row["anomaly"] == -1:
        score += 30
        reasons.append("ML Anomaly")

    if row["Class"] == 1:
        score += 10
        reasons.append("Labelled Fraud")

    return pd.Series([score, ", ".join(reasons)])

df[["risk_score","risk_reasons"]] = df.apply(risk_engine, axis=1)

df["alert"] = (df["risk_score"] >= 40).astype(int)

# ------------------------------------------------
# NAVIGATION
# ------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard","Monitoring","Investigation Queue","Evaluation","STR Generator"]
)

# ------------------------------------------------
# DASHBOARD
# ------------------------------------------------
if page == "Dashboard":

    st.subheader("AML Overview")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Transactions", len(df))
    c2.metric("Alerts", int(df["alert"].sum()))
    c3.metric("Avg Amount", round(df["Amount"].mean(),2))
    c4.metric("Investigate Threshold", round(investigate_threshold,2))

    fig = px.histogram(df, x="Amount", title="Amount Distribution")
    st.plotly_chart(fig, use_container_width=True)

    geo = px.pie(df, names="destination_country", title="Country Exposure")
    st.plotly_chart(geo, use_container_width=True)

# ------------------------------------------------
# MONITORING VIEW
# ------------------------------------------------
elif page == "Monitoring":

    st.subheader("Transaction Monitoring + Explainability")

    cols = ["Amount","risk_score","risk_reasons","amount_risk_flag",
            "behavior_deviation","velocity_flag","anomaly"]

    st.dataframe(df[cols].head(100))

# ------------------------------------------------
# INVESTIGATION QUEUE
# ------------------------------------------------
elif page == "Investigation Queue":

    st.subheader("Investigation Required")

    investigate_df = df[df["alert"] == 1]

    st.metric("Total Investigation Alerts", len(investigate_df))
    st.dataframe(investigate_df.head(100))

# ------------------------------------------------
# EVALUATION METRICS
# ------------------------------------------------
elif page == "Evaluation":

    st.subheader("Model Evaluation")

    report = classification_report(df["Class"], df["alert"], output_dict=True)
    st.json(report)

    cm = confusion_matrix(df["Class"], df["alert"])
    st.write("Confusion Matrix")
    st.write(cm)

# ------------------------------------------------
# STR GENERATOR
# ------------------------------------------------
elif page == "STR Generator":

    st.subheader("Suspicious Transaction Report")

    selected = df[df["alert"] == 1]

    if len(selected) == 0:
        st.warning("No alerts available.")
    else:
        tx = selected.sample(1).iloc[0]

        narrative = f"""
Suspicious Transaction Report

Amount: {tx['Amount']}
Risk Score: {tx['risk_score']}
Reasons: {tx['risk_reasons']}

This transaction exceeds dynamic risk thresholds and triggers AML monitoring alerts.
"""

        st.text_area("Auto Narrative", narrative, height=200)

        st.download_button(
            label="Download STR",
            data=narrative,
            file_name="STR_Report.txt"
        )
