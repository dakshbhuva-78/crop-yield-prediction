import streamlit as st
import pickle
import pandas as pd

# Load files
model   = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))
scaler  = pickle.load(open("scaler.pkl", "rb"))

# Extract dropdown values
crops   = sorted([c.replace("Crop_", "")   for c in columns if c.startswith("Crop_")])
seasons = sorted([c.replace("Season_", "") for c in columns if c.startswith("Season_")])
states  = sorted([c.replace("State_", "")  for c in columns if c.startswith("State_")])

st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾")
st.title("🌾 Crop Yield Predictor")
st.markdown("Fill in the details below to predict the crop yield.")

# Dropdowns
crop   = st.selectbox("🌿 Select Crop", crops)
season = st.selectbox("🗓️ Select Season", seasons)
state  = st.selectbox("📍 Select State", states)

st.markdown("### 📥 Enter Field Details")

col1, col2 = st.columns(2)

with col1:
    area        = st.number_input("🌾 Area (in Hectares)", min_value=0.1, value=1.0, step=0.1,
                                   help="Total cultivated land area in Hectares (ha)")
    rainfall    = st.number_input("🌧️ Annual Rainfall (in mm)", min_value=0.0, value=100.0, step=1.0,
                                   help="Total annual rainfall received in Millimeters (mm)")

with col2:
    fertilizer  = st.number_input("🧪 Fertilizer Used (in kg/ha)", min_value=0.0, value=10.0, step=0.5,
                                   help="Amount of fertilizer used per Hectare in Kilograms (kg/ha)")
    pesticide   = st.number_input("🐛 Pesticide Used (in kg/ha)", min_value=0.0, value=1.0, step=0.1,
                                   help="Amount of pesticide used per Hectare in Kilograms (kg/ha)")

if st.button("🔍 Predict Crop Yield"):

    # Build zero-filled input row
    input_dict = {col: 0 for col in columns}

    # Map inputs to exact column names from training
    field_map = {
        "Area": area,
        "Annual_Rainfall": rainfall,
        "Fertilizer": fertilizer,
        "Pesticide": pesticide
    }

    for col, val in field_map.items():
        if col in input_dict:
            input_dict[col] = val

    # One-hot flags
    for prefix, value in [("Crop", crop), ("Season", season), ("State", state)]:
        col_name = f"{prefix}_{value}"
        if col_name in input_dict:
            input_dict[col_name] = 1

    # Enforce column order
    input_df = pd.DataFrame([input_dict])[columns]

    # Scale and predict
    input_scaled = scaler.transform(input_df)
    prediction   = model.predict(input_scaled)[0]

    # Yield category
    if prediction > 200:
        category = "🟢 High"
    elif prediction > 100:
        category = "🟡 Medium"
    else:
        category = "🔴 Low"

    # Results
    st.markdown("---")
    st.subheader("📊 Prediction Results")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(label="🌱 Predicted Yield", value=f"{prediction:.2f} kg/ha")

    with c2:
        st.metric(label="📦 Yield (Rounded)", value=f"{round(prediction)} kg/ha")

    with c3:
        st.metric(label="📈 Yield Category", value=category)

    st.success(f"🌾 Predicted crop yield for **{crop}** in **{state}** during **{season}** season is **{prediction:.2f} kg/hectare**.")

    # Input Summary
    st.markdown("### 📋 Input Summary")
    summary = {
        "Crop": crop,
        "Season": season,
        "State": state,
        "Area": f"{area} ha",
        "Annual Rainfall": f"{rainfall} mm",
        "Fertilizer": f"{fertilizer} kg/ha",
        "Pesticide": f"{pesticide} kg/ha"
    }
    st.table(pd.DataFrame(summary.items(), columns=["Parameter", "Value"]))