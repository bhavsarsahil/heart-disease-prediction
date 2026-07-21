# import streamlit as st
# import pandas as pd
# import joblib   



# model = joblib.load("Logistic_Regression_Heart.pkl")
# scaler = joblib.load("scaler.pkl")
# expected_columns = joblib.load("columns.pkl")

# st.title("Heart Stroke Prediction By Sahil❤️")
# st.markdown("Provide the following details")

# age = st.slider("Age",18,100,40)
# sex = st.selectbox("Sex",['M','F'])
# chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
# resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80,200,120) 
# cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
# fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
# resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
# max_hr = st.slider("Max Heart Rate", 60, 220, 150)
# exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
# oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
# st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# if st.button("Predict"):

#     # Create a raw input dictionary
#     raw_input = {
#         'Age': age,
#         'RestingBP': resting_bp,
#         'Cholesterol': cholesterol,
#         'FastingBS': fasting_bs,
#         'MaxHR': max_hr,
#         'Oldpeak': oldpeak,
#         'Sex_' + sex: 1,
#         'ChestPainType_' + chest_pain: 1,
#         'RestingECG_' + resting_ecg: 1,
#         'ExerciseAngina_' + exercise_angina: 1,
#         'ST_Slope_' + st_slope: 1
#     }
    
#     input_df = pd.DataFrame([raw_input])

#     # Fill in missing columns with 0s
#     for col in expected_columns:
#         if col not in input_df.columns:
#             input_df[col] = 0
            
#     input_df = input_df[expected_columns]
    
#     scaled_input = scaler.transform(input_df)

#     # Make prediction
#     prediction = model.predict(scaled_input)[0]

#     # Show result
#     if prediction == 1:
#         st.error("⚠️ High Risk of Heart Disease")
#     else:
#         st.success("✅ Low Risk of Heart Disease")
import streamlit as st
import pandas as pd
import joblib
import time

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------------------------
model = joblib.load("Logistic_Regression_Heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# ---------------------------------------------------------------------------
# Custom CSS — gradients, glassmorphism cards, keyframe animations
# ---------------------------------------------------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Animated gradient background */
.stApp {
    background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #1a1a2e);
    background-size: 400% 400%;
    animation: gradientShift 18s ease infinite;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Fade + rise entrance for main block */
.main .block-container {
    animation: fadeUp 0.8s ease-out;
    padding-top: 2rem;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Hero title */
.hero-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
    animation: popIn 0.7s ease-out;
}

.hero-title .title-gradient {
    background: linear-gradient(90deg, #ff6a88, #ff99ac, #ff6a88);
    background-size: 200% auto;
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent;
    animation: shine 4s linear infinite;
}

@keyframes shine {
    to { background-position: 200% center; }
}

@keyframes popIn {
    0% { opacity: 0; transform: scale(0.85); }
    60% { opacity: 1; transform: scale(1.03); }
    100% { transform: scale(1); }
}

.hero-subtitle {
    text-align: center;
    color: #cfd3ec;
    font-size: 1.05rem;
    font-weight: 300;
    margin-bottom: 1.8rem;
    animation: fadeUp 1s ease-out;
}

/* Beating heart emoji */
.beating-heart {
    display: inline-block;
    margin-right: 0.35rem;
    animation: beat 1.15s ease-in-out infinite;
}

@keyframes beat {
    0%, 100% { transform: scale(1); }
    15% { transform: scale(1.25); }
    30% { transform: scale(1); }
    45% { transform: scale(1.2); }
    60% { transform: scale(1); }
}

/* Glass card wrapper for form sections — targets Streamlit's native
   bordered container (st.container(border=True)) so widgets actually
   render INSIDE the styled box instead of alongside an empty div. */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 18px !important;
    padding: 1.2rem 1.4rem 0.4rem 1.4rem;
    margin-bottom: 1.3rem;
    transition: transform 0.25s ease, box-shadow 0.25s ease, border 0.25s ease;
    animation: fadeUp 0.9s ease-out;
}

div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 28px rgba(255, 99, 132, 0.18);
    border: 1px solid rgba(255, 105, 135, 0.35) !important;
}

.section-header {
    color: #ffb6c1;
    font-weight: 600;
    font-size: 1.05rem;
    margin-bottom: 0.6rem;
    letter-spacing: 0.3px;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #24243e 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label {
    color: #e6e6f0 !important;
}

/* Widgets: labels */
label, .stMarkdown p {
    color: #e8e8f2 !important;
}

/* Predict button */
div.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
    background-size: 200% auto;
    color: white;
    border: none;
    border-radius: 14px;
    padding: 0.85rem 1.2rem;
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    box-shadow: 0 6px 20px rgba(255, 65, 108, 0.4);
    transition: all 0.35s ease;
    animation: pulseButton 2.4s ease-in-out infinite;
}

div.stButton > button:hover {
    background-position: right center;
    transform: scale(1.02) translateY(-2px);
    box-shadow: 0 10px 28px rgba(255, 65, 108, 0.55);
}

div.stButton > button:active {
    transform: scale(0.98);
}

@keyframes pulseButton {
    0%, 100% { box-shadow: 0 6px 20px rgba(255, 65, 108, 0.4); }
    50% { box-shadow: 0 6px 30px rgba(255, 65, 108, 0.7); }
}

/* Result cards */
.result-card {
    border-radius: 20px;
    padding: 1.8rem 2rem;
    text-align: center;
    animation: resultPop 0.6s cubic-bezier(0.26, 1.36, 0.44, 1);
    margin-top: 1rem;
}

@keyframes resultPop {
    0% { opacity: 0; transform: scale(0.8) translateY(15px); }
    100% { opacity: 1; transform: scale(1) translateY(0); }
}

.result-high {
    background: linear-gradient(135deg, rgba(255, 65, 108, 0.22), rgba(255, 75, 43, 0.10));
    border: 1px solid rgba(255, 90, 90, 0.45);
    box-shadow: 0 0 40px rgba(255, 65, 108, 0.25);
}

.result-low {
    background: linear-gradient(135deg, rgba(46, 213, 115, 0.20), rgba(0, 210, 211, 0.10));
    border: 1px solid rgba(46, 213, 115, 0.45);
    box-shadow: 0 0 40px rgba(46, 213, 115, 0.25);
}

.result-icon {
    font-size: 3.2rem;
    animation: bounceIn 0.8s ease-out;
}

@keyframes bounceIn {
    0% { transform: scale(0); }
    60% { transform: scale(1.25); }
    100% { transform: scale(1); }
}

.result-heading {
    font-size: 1.6rem;
    font-weight: 700;
    margin-top: 0.4rem;
    color: white;
}

.result-sub {
    color: #d8d8e6;
    font-size: 0.95rem;
    margin-top: 0.3rem;
}

/* Metric chips row */
.chip-row {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    justify-content: center;
    margin-top: 1rem;
}

.chip {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 999px;
    padding: 0.35rem 0.9rem;
    font-size: 0.82rem;
    color: #e6e6f0;
    animation: chipIn 0.5s ease-out backwards;
}

.chip:nth-child(1){animation-delay:0.05s;}
.chip:nth-child(2){animation-delay:0.12s;}
.chip:nth-child(3){animation-delay:0.19s;}
.chip:nth-child(4){animation-delay:0.26s;}
.chip:nth-child(5){animation-delay:0.33s;}

@keyframes chipIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Footer */
.footer-note {
    text-align: center;
    color: #8f8fae;
    font-size: 0.78rem;
    margin-top: 2.5rem;
    animation: fadeUp 1.2s ease-out;
}

hr {
    border-color: rgba(255,255,255,0.08) !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — about panel
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ❤️ About this tool")
    st.markdown(
        "This app uses a **Logistic Regression** model trained on clinical "
        "heart-health data to estimate the likelihood of heart disease from "
        "your inputs."
    )
    st.markdown("---")
    st.markdown("### 🧭 How to use")
    st.markdown(
        "1. Fill in the health details on the right\n"
        "2. Click **Predict**\n"
        "3. Review your estimated risk"
    )
    st.markdown("---")
    st.markdown(
        "<span style='font-size:0.8rem;color:#9a9ac0;'>"
        "⚠️ This tool is for educational purposes only and is not a "
        "substitute for professional medical advice."
        "</span>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='hero-title'><span class='beating-heart'>❤️</span>"
    "<span class='title-gradient'>Heart Stroke Prediction</span></div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='hero-subtitle'>Crafted by Sahil — enter your health "
    "details below for an instant risk estimate</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Input form — organized into glass cards
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2, gap="large")

with col1:
    card1 = st.container(border=True)
    with card1:
        st.markdown("<div class='section-header'>🧍 Personal Details</div>", unsafe_allow_html=True)
        age = st.slider("Age", 18, 100, 40)
        sex = st.selectbox("Sex", ['M', 'F'])
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])

with col2:
    card2 = st.container(border=True)
    with card2:
        st.markdown("<div class='section-header'>💓 Cardiac Details</div>", unsafe_allow_html=True)
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.slider("Max Heart Rate", 60, 220, 150)
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
        oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.write("")
predict_clicked = st.button("🔍 Predict My Risk")

# ---------------------------------------------------------------------------
# Prediction logic
# ---------------------------------------------------------------------------
if predict_clicked:

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    with st.spinner("Analyzing your heart health..."):
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        proba = model.predict_proba(scaled_input)[0]
        time.sleep(0.6)  # small delay so the spinner + reveal feel intentional

    risk_score = proba[1] * 100

    if prediction == 1:
        st.markdown(f"""
        <div class="result-card result-high">
            <div class="result-icon">⚠️</div>
            <div class="result-heading">High Risk of Heart Disease</div>
            <div class="result-sub">Estimated risk score: {risk_score:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(min(int(risk_score), 100))
        st.markdown(
            "<div class='footer-note'>Please consider consulting a "
            "cardiologist for a thorough evaluation.</div>",
            unsafe_allow_html=True,
        )
        st.snow()
    else:
        st.markdown(f"""
        <div class="result-card result-low">
            <div class="result-icon">✅</div>
            <div class="result-heading">Low Risk of Heart Disease</div>
            <div class="result-sub">Estimated risk score: {risk_score:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(min(int(risk_score), 100))
        st.balloons()

    st.markdown(f"""
    <div class="chip-row">
        <div class="chip">🎂 Age: {age}</div>
        <div class="chip">🩸 BP: {resting_bp} mmHg</div>
        <div class="chip">🧪 Cholesterol: {cholesterol} mg/dL</div>
        <div class="chip">💓 Max HR: {max_hr}</div>
        <div class="chip">📉 Oldpeak: {oldpeak}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(
    "<div class='footer-note'>Built with Streamlit · Logistic Regression "
    "Model · Made with ❤️ by Sahil</div>",
    unsafe_allow_html=True,
)
