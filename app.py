# ================================
# AI Personal Insurance Advisor
# ================================

from flask import Flask, render_template, request
import pandas as pd
import os
from model import train_risk_model

# ----------------
# Flask App
# ----------------
app = Flask(__name__)

# ----------------
# Ensure dataset folder exists
# ----------------
os.makedirs("dataset", exist_ok=True)

# ----------------
# Load Insurance Rates (SAFE)
# ----------------
rates_df = pd.read_csv("dataset/insurance_rates.csv")

# Normalize column names (ENTERPRISE FIX)
rates_df.columns = (
    rates_df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("\ufeff", "")
)

# Guarantee correct column name
if "policy_type" not in rates_df.columns:
    rates_df.rename(columns={rates_df.columns[0]: "policy_type"}, inplace=True)

# ----------------
# Train ML Model
# ----------------
model = train_risk_model()

# ----------------
# Primitive Functions
# ----------------
def extractData(form):
    name = form["name"].strip().title()
    age = int(form["age"])
    city = form["city"].strip().title()
    policy = form["policy"].strip().title()
    return name, age, city, policy

def lookupValue(policy_type):
    record = rates_df[rates_df["policy_type"] == policy_type]

    if record.empty:
        raise ValueError("Policy not found")

    return record.iloc[0]["base_rate"], record.iloc[0]["city_risk_factor"]

def calculate(age):
    return round(model.predict([[age]])[0], 2)

def applyFormula(base_rate, risk_score, city_factor):
    return int(base_rate * risk_score * city_factor)

def saveToCSV(data):
    file_path = "dataset/customer_quotes.csv"

    if not os.path.exists(file_path):
        pd.DataFrame(columns=data.keys()).to_csv(file_path, index=False)

    pd.DataFrame([data]).to_csv(
        file_path, mode="a", header=False, index=False
    )

def displayInformation(name, premium):
    return f"{name}, your personalized insurance premium is ₹{premium}"

# ----------------
# Web Route
# ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    premium = None
    message = None

    if request.method == "POST":
        try:
            name, age, city, policy = extractData(request.form)
            base_rate, city_factor = lookupValue(policy)
            risk_score = calculate(age)
            premium = applyFormula(base_rate, risk_score, city_factor)

            saveToCSV({
                "name": name,
                "age": age,
                "city": city,
                "policy_type": policy,
                "risk_score": risk_score,
                "premium": premium
            })

            message = displayInformation(name, premium)

        except Exception as e:
            message = f"Error: {str(e)}"

    return render_template("index.html", premium=premium, message=message)

# ----------------
# Run Server
# ----------------
if __name__ == "__main__":
    app.run(debug=True)
