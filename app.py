import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest

# -------------------------------------
# CONFIG
# -------------------------------------
st.set_page_config(page_title="IFSC AML Monitoring System", layout="wide")
st.title("IFSC Risk-Based AML Monitoring System")

# -------------------------------------
# FILE UPLOAD
# -------------------------------------
uploaded_file = st.file_uploader("Upload creditcard.csv", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # -------------------------------------
    # ADD AML STYLE FIELDS
    # -------------------------------------
    countries = ["India","UAE","UK","Singapore","Cayman Islands","Panama"]
    high_risk = ["Cayman Islands","Panama"]

    df["origin_country"] = np.random.choice(countries,len(df))
    df["destination_country"] = np.random.choice(countries,len(df))

    df["high_risk_country_flag"] = df["destination_country"].apply(
        lambda x: 1 if x in high_risk else 0
    )

    # -------------------------------------
    # DYNAMIC INVESTIGATION THRESHOLD
    # -------------------------------------
    percentile = st.sidebar.slider(
        "Investigation Threshold Percentile",
        80, 99, 95
    )

    investigate_threshold = df["Amount"].quantile(percentile/100)
    high_risk_threshold = df["Amount"].quantile(0.99)

    def amount_risk(amount):

        if amount > high_risk_threshold:
            return "High Risk – Immediate Review"

        elif amount > investigate_threshold:
            return "Higher Risk – Investigate"

        else:
            return "Normal"

    df["amount_risk_flag"] = df["Amount"].apply(amount_risk)

    # -------------------------------------
    # AML RULE ENGINE
    # -------------------------------------
    def aml_rule_engine(row):

        score = 0

        if row["amount_risk_flag"] != "Normal":
            score += 30

        if row["Class"] == 1:
            score += 40

        if row["high_risk_country_flag"] == 1:
            score += 35

        return score

    df["rule_score"] = df.apply(aml_rule_engine,axis=1)

    # -------------------------------------
    # ML ANOMALY DETECTION
    # -------------------------------------
    model = IsolationForest(contamination=0.01,random_state=42)
    df["anomaly"] = model.fit_predict(df[["Amount","Time"]])

    df["alert"] = np.where(
        (df["rule_score"] >= 40) | (df["anomaly"] == -1),
        1,
        0
    )

    # -------------------------------------
    # NAVIGATION
    # -------------------------------------
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard","Transaction Monitoring",
         "Risk Scoring","Investigation Queue",
         "STR Generator"]
    )

    # -------------------------------------
    # DASHBOARD
    # -------------------------------------
    if page == "Dashboard":

        st.subheader("AML Dashboard")

        c1,c2,c3,c4 = st.columns(4)

        c1.metric("Total Transactions",len(df))
        c2.metric("Fraud Labelled",int(df["Class"].sum()))
        c3.metric("AML Alerts",int(df["alert"].sum()))
        c4.metric("Investigation Threshold",round(investigate_threshold,2))

        fig = px.histogram(df,x="Amount")
        st.plotly_chart(fig,use_container_width=True)

    # -------------------------------------
    # MONITORING
    # -------------------------------------
    elif page == "Transaction Monitoring":

        st.subheader("Transaction Monitoring")

        st.dataframe(df.head(100))

    # -------------------------------------
    # RISK SCORING
    # -------------------------------------
    elif page == "Risk Scoring":

        st.subheader("Risk-Based Scoring")

        sample = df.sample(1).iloc[0]

        st.json(sample.to_dict())

        st.success(f"Rule Score: {sample['rule_score']}")
        st.warning(f"Amount Risk: {sample['amount_risk_flag']}")

    # -------------------------------------
    # INVESTIGATION QUEUE
    # -------------------------------------
    elif page == "Investigation Queue":

        st.subheader("Higher Risk – Investigation Required")

        investigate_df = df[
            df["amount_risk_flag"] != "Normal"
        ]

        st.dataframe(investigate_df.head(100))

    # -------------------------------------
    # STR GENERATOR
    # -------------------------------------
    elif page == "STR Generator":

        st.subheader("Suspicious Transaction Report")

        tx_id = st.text_input("Transaction ID")
        reason = st.text_area("Reason for Suspicion")

        if st.button("Generate STR"):

            report = f"""
Suspicious Transaction Report

Transaction ID: {tx_id}

Reason:
{reason}

Generated under IFSC AML Monitoring Framework.
"""

            st.download_button(
                label="Download STR",
                data=report,
                file_name="STR_Report.txt"
            )

else:
    st.info("Upload creditcard.csv to begin AML monitoring.")
