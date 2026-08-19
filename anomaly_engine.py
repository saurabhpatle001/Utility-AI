import pandas as pd


def detect_anomalies(df):

    incidents = []

    latest_time = df["timestamp"].max()

    latest_data = df[
        df["timestamp"] == latest_time
    ].copy()

    for _, row in latest_data.iterrows():

        meter_id = row["meter_id"]

        meter_history = df[
            df["meter_id"] == meter_id
        ]

        historical_data = meter_history[
            meter_history["timestamp"] < latest_time
        ]

        baseline = historical_data["kwh"].mean()

        current_kwh = row["kwh"]

        if baseline == 0:
            continue

        deviation = (
            (current_kwh - baseline)
            / baseline
        ) * 100

        # =================================
        # Consumption Spike
        # =================================

        if deviation > 50:

            incidents.append({
                "meter_id": meter_id,
                "location": row["location"],
                "type": "Consumption Spike",
                "severity": "HIGH",
                "score": min(
                    100,
                    int(deviation)
                ),
                "current_kwh": round(
                    current_kwh,
                    2
                ),
                "baseline_kwh": round(
                    baseline,
                    2
                ),
                "deviation": round(
                    deviation,
                    2
                ),
                "voltage": round(
                    row["voltage"],
                    2
                ),
                "status": row["status"]
            })

        # =================================
        # Voltage Anomaly
        # =================================

        if (
            row["voltage"] > 250
            or
            row["voltage"] < 200
        ):

            incidents.append({
                "meter_id": meter_id,
                "location": row["location"],
                "type": "Voltage Anomaly",
                "severity": "MEDIUM",
                "score": 50,
                "current_kwh": round(
                    current_kwh,
                    2
                ),
                "baseline_kwh": round(
                    baseline,
                    2
                ),
                "deviation": round(
                    deviation,
                    2
                ),
                "voltage": round(
                    row["voltage"],
                    2
                ),
                "status": row["status"]
            })

    return pd.DataFrame(incidents)