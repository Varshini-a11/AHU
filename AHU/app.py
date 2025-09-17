import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# --------------------------
# Load model and dataset
# --------------------------
model = load_model(r"D:\Downloads\AHU\my_trained_model.keras")

SEQ_LEN = 40
FORECAST_STEPS = 5
FEATURES = ["CCALTemp", "HCALTemp"]
UPPER_PERCENTILE = 0.95
LOWER_PERCENTILE = 0.05

df = pd.read_csv(r"D:\Downloads\AHU\Data_Article_Dataset.csv")
df_selected = df[FEATURES].dropna()

# Fit scaler
scaler = MinMaxScaler()
scaler.fit(df_selected.values)

# Default percentile thresholds
ccal_upper = np.percentile(df_selected["CCALTemp"], UPPER_PERCENTILE*100)
hcal_upper = np.percentile(df_selected["HCALTemp"], UPPER_PERCENTILE*100)
ccal_lower = np.percentile(df_selected["CCALTemp"], LOWER_PERCENTILE*100)
hcal_lower = np.percentile(df_selected["HCALTemp"], LOWER_PERCENTILE*100)

# --------------------------
# Streamlit UI
# --------------------------
st.title("🌡️ AHU Forecast Dashboard")
st.markdown("### 📉 40 Steps Input → Next 5 Steps Prediction")

# --------------------------
# Sidebar: User-defined thresholds
# --------------------------
st.sidebar.header("⚙️ Threshold Settings")
ccal_upper = float(st.sidebar.text_input("CCAL Upper Threshold", value=str(ccal_upper)))
ccal_lower = float(st.sidebar.text_input("CCAL Lower Threshold", value=str(ccal_lower)))
hcal_upper = float(st.sidebar.text_input("HCAL Upper Threshold", value=str(hcal_upper)))
hcal_lower = float(st.sidebar.text_input("HCAL Lower Threshold", value=str(hcal_lower)))

# --------------------------
# User input sequence
# --------------------------
st.subheader("✍️ Enter 40 CCAL values separated by commas")
ccal_input = st.text_area("CCAL values", value=",".join(["0"]*SEQ_LEN))
st.subheader("✍️ Enter 40 HCAL values separated by commas")
hcal_input = st.text_area("HCAL values", value=",".join(["0"]*SEQ_LEN))

if st.button("🔮 Forecast Next 5 Steps"):
    try:
        # Convert input to float lists
        ccal_list = [float(x.strip()) for x in ccal_input.split(",")]
        hcal_list = [float(x.strip()) for x in hcal_input.split(",")]

        if len(ccal_list) != SEQ_LEN or len(hcal_list) != SEQ_LEN:
            st.error(f"⚠️ Please enter exactly {SEQ_LEN} values for each input.")
        else:
            # Prepare input sequence
            input_seq = np.array([list(x) for x in zip(ccal_list, hcal_list)])  # (40,2)

            # Scale input
            input_scaled = scaler.transform(input_seq).reshape(1, SEQ_LEN, len(FEATURES))

            # Forecast next 5 steps
            seq = input_scaled.copy()
            predictions_scaled = []

            for _ in range(FORECAST_STEPS):
                pred_scaled = model.predict(seq, verbose=0)
                pred_values = pred_scaled[0] if pred_scaled.ndim == 2 else pred_scaled[0,0]
                predictions_scaled.append(pred_values)
                seq = np.vstack([seq[0,1:], pred_values]).reshape(1, SEQ_LEN, len(FEATURES))

            predictions_scaled = np.array(predictions_scaled)
            predictions = scaler.inverse_transform(predictions_scaled)

            # --------------------------
            # Check failures
            # --------------------------
            fail_steps = {f: [] for f in FEATURES}
            for step, row in enumerate(predictions):
                if row[0] > ccal_upper or row[0] < ccal_lower:
                    fail_steps["CCALTemp"].append(step)
                if row[1] > hcal_upper or row[1] < hcal_lower:
                    fail_steps["HCALTemp"].append(step)

            steps_total = SEQ_LEN + FORECAST_STEPS

            # --------------------------
            # CCAL Graph
            # --------------------------
            fig_ccal, ax_ccal = plt.subplots(figsize=(10,4))
            full_ccal = np.concatenate([input_seq[:, 0], predictions[:, 0]])
            ax_ccal.plot(range(steps_total), full_ccal, color="blue", label="CCAL (Input + Forecast)")

            ax_ccal.scatter(range(SEQ_LEN), input_seq[:, 0], color="blue", s=50, marker='o', label="CCAL Input Points")
            ax_ccal.scatter(range(SEQ_LEN, steps_total), predictions[:, 0], color="orange", s=50, marker='^', label="CCAL Forecast Points")

            ax_ccal.hlines([ccal_upper, ccal_lower], xmin=0, xmax=steps_total-1,
                           colors='green', linestyles='dashed', label="CCAL Thresholds")

            if fail_steps["CCALTemp"]:
                ax_ccal.scatter([SEQ_LEN + s for s in fail_steps["CCALTemp"]], 
                                predictions[fail_steps["CCALTemp"], 0],
                                color='red', s=80, marker='x', label="CCAL Failure")

            ax_ccal.set_title("📊 CCAL Input + Forecast")
            ax_ccal.set_xlabel("Steps")
            ax_ccal.set_ylabel("CCAL Values")
            ax_ccal.grid(True)

            # Legend outside graph
            ax_ccal.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            st.pyplot(fig_ccal)

            # --------------------------
            # HCAL Graph
            # --------------------------
            fig_hcal, ax_hcal = plt.subplots(figsize=(10,4))
            full_hcal = np.concatenate([input_seq[:, 1], predictions[:, 1]])
            ax_hcal.plot(range(steps_total), full_hcal, color="red", label="HCAL (Input + Forecast)")

            ax_hcal.scatter(range(SEQ_LEN), input_seq[:, 1], color="red", s=50, marker='o', label="HCAL Input Points")
            ax_hcal.scatter(range(SEQ_LEN, steps_total), predictions[:, 1], color="orange", s=50, marker='^', label="HCAL Forecast Points")

            ax_hcal.hlines([hcal_upper, hcal_lower], xmin=0, xmax=steps_total-1,
                           colors='purple', linestyles='dashed', label="HCAL Thresholds")

            if fail_steps["HCALTemp"]:
                ax_hcal.scatter([SEQ_LEN + s for s in fail_steps["HCALTemp"]], 
                                predictions[fail_steps["HCALTemp"], 1],
                                color='black', s=80, marker='x', label="HCAL Failure")

            ax_hcal.set_title("📊 HCAL Input + Forecast")
            ax_hcal.set_xlabel("Steps")
            ax_hcal.set_ylabel("HCAL Values")
            ax_hcal.grid(True)

            # Legend outside graph
            ax_hcal.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            st.pyplot(fig_hcal)

    except Exception as e:
        st.error(f"❌ Error: {e}")
