import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()


def get_ai_investigation(incident):

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="openai/gpt-oss-20b",
        temperature=0.2
    )


    prompt = f"""
You are an AI Utility Operations Analyst.

Analyze the following smart meter incident.

Incident information:

Meter ID: {incident.get("meter_id")}
Location: {incident.get("location")}

Incident Type: {incident.get("incident")}
Severity: {incident.get("severity")}
Risk Score: {incident.get("score")}

Current Consumption:
{incident.get("kwh")} kWh

Baseline Consumption:
{incident.get("baseline")} kWh

Deviation:
{incident.get("deviation"):.1f}%

Voltage:
{incident.get("voltage")} V

Current:
{incident.get("current")} A

Status:
{incident.get("status")}


Provide a concise operational investigation.

Use exactly these sections:

## 1. What Happened?

Explain the detected anomaly.

## 2. Why It Matters

Explain the operational significance.

## 3. Possible Causes

List 3-4 realistic possible causes.

## 4. Recommended Checks

Give practical checks that a utility operator
or field engineer could perform.

## 5. Recommended Action

Give a clear next action.

Important:

- Do not invent sensor readings.
- Use only the information provided.
- Keep the explanation practical.
- This is an operational decision-support system,
  not an automatic control system.
"""


    response = llm.invoke(prompt)

    return response.content