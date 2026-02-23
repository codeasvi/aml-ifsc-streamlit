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
uploaded_file = st.file_uploader(
    "Upload Credit Card Dataset (creditcard.csv)",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # ------------------------------
    # ADD AML STYLE FIELDS
    # ------------------------------
    countries = ["India","UAE","UK","Singapore","Cayman Islands","Panama"]
    high_risk = ["Cayman Islands","Panama"]

    df["origin_country"] = np.random.choice(countries,len(df))
    df["destination_country"] = np.random.choice(countries,len(df))

    df["high_risk_country_flag"] = df["destination_country"].apply(
        lambda x: 1 if x in high_risk else 0
    )

    # ------------------------------
    # AML RULE ENGINE
    # ------------------------------
    def aml_rule_engine(row):

        score = 0

        if row["Amount"] > 2000:
            score += 25

        if row["Class"] == 1:
            score += 40

        if row["high_risk_country_flag"] == 1:
            score += 35

        return score

    df["rule_score"] = df.apply(aml_rule_engine,axis=1)

    # ------------------------------
    # ML ANOMALY DETECTION
    # ------------------------------
    model = IsolationForest(contamination=0.01,random_state=42)
    df["anomaly"] = model.fit_predict(df[["Amount","Time"]])

    df["alert"] = np.where(
        (df["rule_score"] >= 40) | (df["anomaly"] == -1),
        1,
        0
    )

    # ------------------------------
    # NAVIGATION
    # ------------------------------
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard","Transaction Monitoring",
         "Risk Scoring","Alert Queue","STR Generator"]
    )

    # ------------------------------
    # DASHBOARD
    # ------------------------------
    if page == "Dashboard":

        st.subheader("AML Dashboard")

        c1,c2,c3,c4 = st.columns(4)

        c1.metric("Total Transactions",len(df))
        c2.metric("Fraud Labelled",int(df["Class"].sum()))
        c3.metric("AML Alerts",int(df["alert"].sum()))
        c4.metric("Average Amount",round(df["Amount"].mean(),2))

        fig1 = px.histogram(df,x="Amount",
                            title="Transaction Amount Distribution")
        st.plotly_chart(fig1,use_container_width=True)

        fig2 = px.pie(df,names="destination_country",
                      title="Country Exposure")
        st.plotly_chart(fig2,use_container_width=True)

    # ------------------------------
    # MONITORING
    # ------------------------------
    elif page == "Transaction Monitoring":

        st.subheader("Transaction Monitoring")
        st.dataframe(df.head(100))

    # ------------------------------
    # RISK SCORING
    # ------------------------------
    elif page == "Risk Scoring":

        st.subheader("Risk-Based Scoring")

        sample = df.sample(1).iloc[0]
        risk_score = sample["rule_score"]

        if risk_score >= 60:
            level = "High"
        elif risk_score >= 30:
            level = "Medium"
        else:
            level = "Low"

        st.json(sample.to_dict())
        st.success(f"Risk Score: {risk_score}")
        st.warning(f"Risk Level: {level}")

    # ------------------------------
    # ALERT QUEUE
    # ------------------------------
    elif page == "Alert Queue":

        st.subheader("AML Alerts")

        alerts = df[df["alert"] == 1]

        st.metric("Total Alerts",len(alerts))
        st.dataframe(alerts.head(100))

    # ------------------------------
    # STR GENERATOR
    # ------------------------------
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
    st.info("Please upload creditcard.csv to start AML analysis.")
