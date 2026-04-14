import streamlit as st
import numpy as np
import joblib
import pandas as pd
import altair as alt
from io import BytesIO
import base64


# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Solar Power Generation Prediction",
    layout="wide"
)

# --------------------------------------------------
# Background Image 
# --------------------------------------------------
def set_bg(image_path):
    with open(image_path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.75);
            z-index: -1;
        }}

        .card {{
            background: rgba(0,0,0,0.65);
            padding: 16px;
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.2);
        }}

        .predicted-power {{
            margin-top: 10px;
            margin-bottom: 16px;
            padding: 18px 22px;
            border-radius: 16px;
            background: linear-gradient(
                135deg,
                rgba(0, 180, 120, 0.35),
                rgba(0, 120, 80, 0.20)
            );
            border: 1px solid rgba(0, 255, 180, 0.55);
            box-shadow: 0 0 22px rgba(0, 255, 180, 0.35);
            font-size: 28px;
            font-weight: 700;
            color: #eafff4;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg("background.png")

# --------------------------------------------------
# Load model
# --------------------------------------------------
model = joblib.load("model.pkl")

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("☀️ Solar Power Generation Prediction")
st.write("Predict solar energy output based on environmental inputs via data.")
st.markdown("---")

left_col, right_col = st.columns(2, gap="large")

# ==================================================
# LEFT SIDE – INPUTS + FEATURE IMPORTANCE
# ==================================================
with left_col:
    st.subheader("Input Environmental Conditions")

    distance = st.slider("Distance to Solar Noon", 0.0, 1.5, 0.5, 0.01)
    temperature = st.selectbox("Temperature (°C)", list(range(0, 51)), index=25)

    c1, c2 = st.columns(2)
    with c1:
        wind_direction = st.selectbox("Wind Direction (degrees)", list(range(0, 361)))
    with c2:
        wind_speed = st.selectbox(
            "Wind Speed (m/s)",
            [round(x, 1) for x in np.arange(0, 20.1, 0.5)],
            index=10
        )

    sky_cover = st.radio("Sky Cover", ["Clear", "Partly Cloudy", "Cloudy"], horizontal=True)

    c3, c4 = st.columns(2)
    with c3:
        visibility = st.selectbox(
            "Visibility (km)",
            [round(x, 1) for x in np.arange(0, 20.1, 0.5)],
            index=20
        )
    with c4:
        humidity = st.selectbox("Humidity (%)", list(range(0, 101)), index=50)

    c5, c6 = st.columns(2)
    with c5:
        avg_wind_speed = st.selectbox(
            "Average Wind Speed (period)",
            [round(x, 1) for x in np.arange(0, 50.1, 0.5)],
            index=60
        )
    with c6:
        avg_pressure = st.selectbox("Average Pressure (period)", list(range(950, 1051)), index=60)

    # -------- Feature Importance --------
    st.markdown("### 🔍 Feature Importance")

    importance = model.feature_importances_
    features = [
        "Solar Noon Distance", "Temperature", "Wind Direction",
        "Wind Speed", "Sky Cover", "Visibility",
        "Humidity", "Avg Wind Speed", "Avg Pressure"
    ]

    fi_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    fi_chart = alt.Chart(fi_df).mark_bar(size=30).encode(
        x=alt.X("Feature:N", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Importance:Q"),
        color=alt.Color("Feature:N", legend=None)
    )

    st.altair_chart(fi_chart, use_container_width=True)

# ==================================================
# RIGHT SIDE – OUTPUT
# ==================================================
with right_col:
    st.subheader("Predicted Solar Power Generation")

    # BUTTON 
    predict_btn = st.button("Predict Power Generated")

    if predict_btn:
        sky_map = {"Clear": 0, "Partly Cloudy": 2, "Cloudy": 4}

        input_data = np.array([[
            distance,
            temperature,
            wind_direction,
            wind_speed,
            sky_map[sky_cover],
            visibility,
            humidity,
            avg_wind_speed,
            avg_pressure
        ]])

        prediction = model.predict(input_data)[0]

        st.markdown(
            f"""
            <div class="predicted-power">
                Predicted Power: {prediction:.2f} Joules
            </div>
            """,
            unsafe_allow_html=True
        )

        c7, c8 = st.columns(2)
        with c7:
            st.markdown("<div class='card'>Model Used<br><b>Random Forest</b></div>", unsafe_allow_html=True)
        with c8:
            st.markdown("<div class='card'>Accuracy<br><b>R² = 0.91</b></div>", unsafe_allow_html=True)

        st.markdown("### 📊 Visualization Insights")

        summary_df = pd.DataFrame({
            "Value": [humidity, distance, temperature, visibility, wind_speed]
        }, index=[
            "Humidity (%)",
            "Solar Noon Distance",
            "Temperature (°C)",
            "Visibility (km)",
            "Wind Speed (m/s)"
        ])

        chart_df = summary_df.reset_index()
        chart_df.columns = ["Feature", "Value"]

        chart = alt.Chart(chart_df).mark_bar(size=34).encode(
            x=alt.X("Feature:N", axis=alt.Axis(labelAngle=0)),
            y="Value:Q",
            color=alt.Color("Feature:N", legend=None),
            tooltip=["Feature", "Value"]
        )

        labels = chart.mark_text(dy=-8, color="white").encode(
            text=alt.Text("Value:Q", format=".2f")
        )

        st.altair_chart(chart + labels, use_container_width=True)

        report_df = pd.DataFrame({
            "Parameter": chart_df["Feature"],
            "Value": chart_df["Value"]
        })

        buffer = BytesIO()
        report_df.to_csv(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            "Download Prediction Report",
            data=buffer,
            file_name="solar_power_prediction_report.csv",
            mime="text/csv"
        )
