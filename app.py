import streamlit as st
import plotly.express as px

from data_loader import load_meter_data
from anomaly_engine import detect_anomalies
from ai_service import get_ai_investigation
from rag_service import answer_question


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Utility Sentinel AI",
    page_icon="⚡",
    layout="wide"
)


# ==========================================
# HEADER
# ==========================================

st.title("⚡ Utility Sentinel AI")

st.caption(
    "Smart Utility Operations & Incident Intelligence Platform"
)


# ==========================================
# LOAD DATA
# ==========================================

df = load_meter_data()

incidents = detect_anomalies(df)


# ==========================================
# KPI CALCULATIONS
# ==========================================

total_meters = df["meter_id"].nunique()

online_meters = df[
    df["status"] == "online"
]["meter_id"].nunique()

active_incidents = len(incidents)

high_risk_incidents = len(
    incidents[
        incidents["severity"] == "HIGH"
    ]
)


# ==========================================
# KPI CARDS
# ==========================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Meters",
    total_meters
)

col2.metric(
    "Online Meters",
    online_meters
)

col3.metric(
    "Active Incidents",
    active_incidents
)

col4.metric(
    "High Risk",
    high_risk_incidents
)


st.divider()


# ==========================================
# APPLICATION TABS
# ==========================================

tab1, tab2, tab3 = st.tabs([
    "📊 Command Center",
    "🚨 Incident Center",
    "📚 Utility Knowledge"
])


# ==========================================================
# COMMAND CENTER
# ==========================================================

with tab1:

    st.subheader(
        "Energy Consumption Overview"
    )

    consumption = (
        df.groupby("timestamp")["kwh"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        consumption,
        x="timestamp",
        y="kwh",
        markers=True,
        title="Total Energy Consumption"
    )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Energy (kWh)"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


    # ======================================
    # METER OVERVIEW
    # ======================================

    st.subheader(
        "Meter Overview"
    )

    meter_summary = (
        df.groupby("meter_id")
        .agg(
            Average_kWh=("kwh", "mean"),
            Average_Voltage=("voltage", "mean"),
            Average_Current=("current", "mean")
        )
        .reset_index()
    )

    st.dataframe(
        meter_summary,
        width="stretch",
        hide_index=True
    )


# ==========================================================
# INCIDENT CENTER
# ==========================================================

with tab2:

    st.subheader(
        "🚨 Active Utility Incidents"
    )

    if incidents.empty:

        st.success(
            "No anomalies detected."
        )

    else:

        # ==================================
        # INCIDENT TABLE
        # ==================================

        st.dataframe(
            incidents[
                [
                    "meter_id",
                    "location",
                    "type",
                    "severity",
                    "score"
                ]
            ],
            width="stretch",
            hide_index=True
        )


        st.divider()


        # ==================================
        # SELECT INCIDENT
        # ==================================

        st.subheader(
            "🔎 Investigate an Incident"
        )

        incident_options = (
            incidents["meter_id"]
            .astype(str)
            .tolist()
        )

        selected_meter = st.selectbox(
            "Select a meter to investigate:",
            incident_options
        )


        # ==================================
        # SELECTED INCIDENT
        # ==================================

        selected_incident = incidents[
            incidents["meter_id"].astype(str)
            == selected_meter
        ].iloc[0]


        # ==================================
        # GET METER DATA
        # ==================================

        meter_data = df[
            df["meter_id"].astype(str)
            == selected_meter
        ].copy()

        meter_data = meter_data.sort_values(
            "timestamp"
        )

        latest_reading = meter_data.iloc[-1]


        current_kwh = float(
            latest_reading["kwh"]
        )

        baseline_kwh = float(
            meter_data["kwh"].mean()
        )

        voltage = float(
            latest_reading["voltage"]
        )

        current = float(
            latest_reading["current"]
        )

        status = latest_reading["status"]


        if baseline_kwh != 0:

            deviation = (
                (current_kwh - baseline_kwh)
                / baseline_kwh
            ) * 100

        else:

            deviation = 0


        # ==================================
        # INCIDENT DETAILS
        # ==================================

        st.subheader(
            f"📍 Incident Details — {selected_meter}"
        )


        d1, d2, d3, d4 = st.columns(4)

        d1.metric(
            "Incident Type",
            selected_incident["type"]
        )

        d2.metric(
            "Severity",
            selected_incident["severity"]
        )

        d3.metric(
            "Risk Score",
            selected_incident["score"]
        )

        d4.metric(
            "Status",
            status
        )


        # ==================================
        # INCIDENT EVIDENCE
        # ==================================

        st.subheader(
            "🔎 Incident Evidence"
        )


        e1, e2, e3 = st.columns(3)

        e1.metric(
            "Current Usage",
            f"{current_kwh:.2f} kWh"
        )

        e2.metric(
            "Baseline",
            f"{baseline_kwh:.2f} kWh"
        )

        e3.metric(
            "Deviation",
            f"{deviation:+.1f}%"
        )


        e4, e5, e6 = st.columns(3)

        e4.metric(
            "Voltage",
            f"{voltage:.1f} V"
        )

        e5.metric(
            "Current",
            f"{current:.2f} A"
        )

        e6.metric(
            "Meter Status",
            status
        )


        st.write(
            f"**Location:** "
            f"{selected_incident['location']}"
        )


        # ==================================
        # METER CONSUMPTION HISTORY
        # ==================================

        st.subheader(
            "📈 Meter Consumption History"
        )

        meter_history = meter_data.sort_values(
            "timestamp"
        )

        fig_meter = px.line(
            meter_history,
            x="timestamp",
            y="kwh",
            markers=True,
            title=f"{selected_meter} Consumption Trend"
        )

        fig_meter.update_layout(
            xaxis_title="Time",
            yaxis_title="Energy (kWh)"
        )

        st.plotly_chart(
            fig_meter,
            width="stretch"
        )


        st.divider()


        # ==================================
        # AI INVESTIGATION
        # ==================================

        st.subheader(
            "🤖 AI Incident Investigator"
        )

        st.write(
            "Use AI to analyze the incident, "
            "identify possible causes and "
            "recommend operational checks."
        )


        if st.button(
            "🔍 Investigate Incident with AI",
            type="primary"
        ):

            incident_for_ai = {

                "meter_id":
                    selected_meter,

                "location":
                    selected_incident["location"],

                "incident":
                    selected_incident["type"],

                "severity":
                    selected_incident["severity"],

                "score":
                    selected_incident["score"],

                "kwh":
                    current_kwh,

                "baseline":
                    baseline_kwh,

                "deviation":
                    deviation,

                "voltage":
                    voltage,

                "current":
                    current,

                "status":
                    status
            }


            with st.spinner(
                "AI is investigating the incident..."
            ):

                try:

                    investigation = (
                        get_ai_investigation(
                            incident_for_ai
                        )
                    )


                    st.success(
                        "AI investigation completed."
                    )


                    st.markdown(
                        "### 🧠 AI Investigation"
                    )

                    st.markdown(
                        investigation
                    )


                    st.info(
                        "⚠️ This system provides "
                        "decision support. Final "
                        "operational decisions should "
                        "be made by qualified utility "
                        "personnel."
                    )


                except Exception as e:

                    st.error(
                        "Unable to connect to the AI service."
                    )

                    st.exception(e)


# ==========================================================
# UTILITY KNOWLEDGE / RAG
# ==========================================================

with tab3:

    st.subheader(
        "📚 Utility Knowledge Assistant"
    )

    st.write(
        "Ask questions about utility operating "
        "procedures using the approved Utility SOP."
    )

    st.caption(
        "Knowledge source: docs/utility_sop.txt"
    )


    st.divider()


    # ======================================
    # QUESTION INPUT
    # ======================================

    question = st.text_area(
        "Ask the Utility Knowledge Assistant",
        placeholder=(
            "Example: What should a field engineer "
            "check when a smart meter stops communicating?"
        ),
        height=100
    )


    # ======================================
    # ASK BUTTON
    # ======================================

    if st.button(
        "🔎 Ask Utility Knowledge Assistant",
        type="primary"
    ):

        if not question.strip():

            st.warning(
                "Please enter a question first."
            )

        else:

            with st.spinner(
                "Searching the Utility SOP and generating an answer..."
            ):

                try:

                    answer = answer_question(
                        question.strip()
                    )


                    st.success(
                        "Answer generated from Utility SOP."
                    )


                    st.markdown(
                        "### 🤖 Utility Knowledge Answer"
                    )

                    st.markdown(
                        answer
                    )


                    st.info(
                        "⚠️ This knowledge assistant "
                        "provides decision support based "
                        "on the available utility SOP. "
                        "Operational actions must be "
                        "validated by qualified utility "
                        "personnel."
                    )


                except Exception as e:

                    st.error(
                        "Unable to generate the knowledge answer."
                    )

                    st.exception(e)