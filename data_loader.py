import pandas as pd
import numpy as np


def generate_demo_data():

    np.random.seed(42)

    meter_ids = [
        f"MTR-{1000 + i}"
        for i in range(1, 26)
    ]

    locations = [
        "Zone A",
        "Zone B",
        "Zone C",
        "Zone D",
        "Zone E"
    ]

    timestamps = pd.date_range(
        end=pd.Timestamp.now().floor("h"),
        periods=24,
        freq="h"
    )

    records = []

    for meter_id in meter_ids:

        location = np.random.choice(locations)

        baseline = np.random.uniform(
            3.0,
            6.0
        )

        for timestamp in timestamps:

            kwh = max(
                0.5,
                np.random.normal(
                    baseline,
                    0.5
                )
            )

            voltage = np.random.normal(
                230,
                3
            )

            current = np.random.normal(
                6,
                1
            )

            power_factor = np.random.uniform(
                0.90,
                0.99
            )

            records.append({
                "timestamp": timestamp,
                "meter_id": meter_id,
                "location": location,
                "kwh": round(kwh, 2),
                "voltage": round(voltage, 2),
                "current": round(current, 2),
                "power_factor": round(
                    power_factor,
                    2
                ),
                "status": "online",
                "last_seen": timestamp
            })

    df = pd.DataFrame(records)

    # =================================
    # DEMO ANOMALY 1
    # Consumption Spike
    # =================================

    latest_time = df["timestamp"].max()

    df.loc[
        (df["meter_id"] == "MTR-1007") &
        (df["timestamp"] == latest_time),
        "kwh"
    ] = 8.9

    # =================================
    # DEMO ANOMALY 2
    # Voltage Anomaly
    # =================================

    df.loc[
        (df["meter_id"] == "MTR-1018") &
        (df["timestamp"] == latest_time),
        "voltage"
    ] = 257

    return df


def load_meter_data():

    return generate_demo_data()