import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import streamlit as st
import joblib
import pandas as pd
from actuarial_calculator import (
    check_values, net_premium, gross_premium, mathematical_reserve,
    surrender_value, calculate_k, calculate_delta,
)
from training_data_split import FEATURE_COLS

@st.cache_resource
def load_models():
    linreg = joblib.load("models/linear_regression.joblib")
    tree = joblib.load("models/decision_tree.joblib")
    return linreg, tree

linreg, tree = load_models()


st.title("Life Insurance Surrender Value Calculator")

unit = st.radio("Enter age and duration in:", ["Years", "Months"])

if unit == "Years":
    age_years = st.number_input("Entry age (years)", min_value=18, max_value=65, value=30)
    duration_years = st.number_input("Policy duration (years)", min_value=2, max_value=35, value=10)
    x = age_years * 12
    n = duration_years * 12
else:
    x = st.number_input("Entry age (months)", min_value=216, max_value=780, value=360)
    n = st.number_input("Policy duration (months)", min_value=24, max_value=420, value=120)

t = st.number_input("Elapsed time t (months)", min_value=0, max_value=int(n), value=0)

pair_options = {
    "OIS=1000 / OID=5000": (1000, 5000),
    "OIS=1000 / OID=4000": (1000, 4000),
    "OIS=1000 / OID=3000": (1000, 3000),
    "OIS=1000 / OID=2000": (1000, 2000),
    "OIS=1000 / OID=1000": (1000, 1000),
    "OIS=2000 / OID=1000": (2000, 1000),
    "OIS=3000 / OID=1000": (3000, 1000),
    "OIS=4000 / OID=1000": (4000, 1000),
    "OIS=5000 / OID=1000": (5000, 1000),
}
choice = st.selectbox("OIS/OID ratio", list(pair_options.keys()))
OIS, OID = pair_options[choice]

if st.button("Calculate"):
    try:
        check_values(x, n, t, OIS, OID)
    except ValueError as e:
        st.error(f"Invalid input: {e}")
    else:
        net = net_premium(x, n, OIS, OID)
        gross = gross_premium(x, n, OIS, OID)
        reserve = mathematical_reserve(x, n, t, OIS, OID)
        surrender = surrender_value(x, n, t, OIS, OID)

        k = calculate_k(OIS, OID)
        delta = calculate_delta(k)

        features = pd.DataFrame([{
            "x": x, "n": n, "t": t, "OIS": OIS, "OID": OID, "k": k, "delta": delta
        }])[FEATURE_COLS]

        linreg_pred = linreg.predict(features)[0]
        tree_pred = tree.predict(features)[0]

        st.subheader("Actuarial calculation")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Net premium", f"{net:.2f}")
        with col2:
            st.metric("Gross premium", f"{gross:.2f}")
        with col3:
            st.metric("Mathematical reserve", f"{reserve:.2f}")
        with col4:
            st.metric("Surrender value", f"{surrender:.2f}")

        st.subheader("Comparison with ML estimates")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Linear regression", f"{linreg_pred:.2f}",
                      delta=f"{linreg_pred - surrender:.2f}")
            st.metric("Absolute deviation of surrender value", f"{abs(linreg_pred - surrender):.2f}")
        with col2:
            st.metric("Decision tree", f"{tree_pred:.2f}",
                      delta=f"{tree_pred - surrender:.2f}")
            st.metric("Absolute deviation of surrender value", f"{abs(tree_pred - surrender):.2f}")