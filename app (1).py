import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Renewable Energy Adoption Predictor",
    page_icon="🌱",
    layout="centered"
)

# ----------------------------
# Glassmorphic UI styling
# ----------------------------
st.markdown(
    """
    <style>
    /* Gradient backdrop behind everything */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 45%, #2c5364 100%);
        background-attachment: fixed;
    }

    /* Floating soft color blobs for depth (purely decorative) */
    .stApp::before {
        content: "";
        position: fixed;
        top: -10%;
        left: -10%;
        width: 45%;
        height: 45%;
        background: radial-gradient(circle, rgba(46,204,113,0.35) 0%, rgba(46,204,113,0) 70%);
        z-index: 0;
        pointer-events: none;
    }
    .stApp::after {
        content: "";
        position: fixed;
        bottom: -15%;
        right: -10%;
        width: 50%;
        height: 50%;
        background: radial-gradient(circle, rgba(52,152,219,0.35) 0%, rgba(52,152,219,0) 70%);
        z-index: 0;
        pointer-events: none;
    }

    /* Main content container turned into a glass panel */
    .block-container {
        position: relative;
        z-index: 1;
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(18px) saturate(160%);
        -webkit-backdrop-filter: blur(18px) saturate(160%);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 20px;
        padding: 2.5rem 2.5rem 2rem 2.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
        margin-top: 2rem;
    }

    /* Text colors for contrast on dark glass */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, span {
        color: #f2f6f5 !important;
    }

    /* Glassy input widgets */
    div[data-baseweb="input"], .stNumberInput input, .stTextInput input {
        background: rgba(255, 255, 255, 0.12) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        color: #ffffff !important;
    }

    /* Glassy expanders */
    .streamlit-expanderHeader, details {
        background: rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        color: #f2f6f5 !important;
    }

    /* Glassy file uploader */
    section[data-testid="stFileUploaderDropzone"] {
        background: rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        border: 1px dashed rgba(255, 255, 255, 0.3) !important;
    }

    /* Primary button styled as frosted pill with green accent */
    .stButton > button {
        background: linear-gradient(135deg, rgba(46,204,113,0.85), rgba(39,174,96,0.85));
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 14px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        backdrop-filter: blur(6px);
        box-shadow: 0 4px 18px rgba(46, 204, 113, 0.35);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(46, 204, 113, 0.45);
        color: #ffffff;
    }

    /* Dataframe / table glass wrapper */
    div[data-testid="stDataFrame"] {
        background: rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 0.4rem;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }

    /* Divider tweak */
    hr {
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Load model & scaler (cached so they load only once)
# ----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("Best_Renewable_Energy_Adoption_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

FEATURES = ["carbon_emissions", "energy_output", "renewability_index", "cost_efficiency"]

# ----------------------------
# UI
# ----------------------------
st.title("🌱 Renewable Energy Adoption Predictor")
st.write(
    "This app uses a trained **Decision Tree Classifier** to predict whether a "
    "renewable energy project is likely to be **adopted**, based on key metrics."
)

st.subheader("Enter project details")

col1, col2 = st.columns(2)
with col1:
    carbon_emissions = st.number_input("Carbon Emissions", value=0.0, format="%.4f")
    energy_output = st.number_input("Energy Output", value=0.0, format="%.4f")
with col2:
    renewability_index = st.number_input("Renewability Index", value=0.0, format="%.4f")
    cost_efficiency = st.number_input("Cost Efficiency", value=0.0, format="%.4f")

if st.button("Predict Adoption", type="primary"):
    input_df = pd.DataFrame(
        [[carbon_emissions, energy_output, renewability_index, cost_efficiency]],
        columns=FEATURES
    )

    # Scale inputs the same way training data was scaled
    input_scaled = scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled, columns=FEATURES)

    prediction = model.predict(input_scaled_df)[0]
    proba = model.predict_proba(input_scaled_df)[0]

    label = "✅ Adoption" if prediction == 1 else "❌ Non-Adoption"
    accent = "rgba(46,204,113,0.28)" if prediction == 1 else "rgba(231,76,60,0.28)"

    st.markdown(
        f"""
        <div style="
            background: {accent};
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.25);
            border-radius: 16px;
            padding: 1.2rem 1.5rem;
            margin-top: 1rem;
            text-align: center;
        ">
            <h3 style="margin:0;">Prediction: {label}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("**Prediction probabilities:**")
    proba_df = pd.DataFrame(
        {"Class": ["Non-Adoption", "Adoption"], "Probability": proba}
    )
    st.bar_chart(proba_df.set_index("Class"))

st.divider()

# ----------------------------
# Optional: show the decision tree
# ----------------------------
with st.expander("🌳 View the Decision Tree structure"):
    fig, ax = plt.subplots(figsize=(14, 8))
    plot_tree(
        model,
        feature_names=FEATURES,
        class_names=["Non-Adoption", "Adoption"],
        filled=True,
        rounded=True,
        ax=ax
    )
    st.pyplot(fig)

# ----------------------------
# Optional: batch prediction via CSV upload
# ----------------------------
with st.expander("📁 Predict from a CSV file"):
    st.write(f"Upload a CSV with columns: {', '.join(FEATURES)}")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        if all(col in batch_df.columns for col in FEATURES):
            scaled_batch = scaler.transform(batch_df[FEATURES])
            preds = model.predict(scaled_batch)
            batch_df["Prediction"] = np.where(preds == 1, "Adoption", "Non-Adoption")
            st.dataframe(batch_df)
            st.download_button(
                "Download predictions as CSV",
                batch_df.to_csv(index=False).encode("utf-8"),
                "predictions.csv",
                "text/csv"
            )
        else:
            st.error(f"CSV must contain these columns: {', '.join(FEATURES)}")
