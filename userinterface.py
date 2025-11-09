import pandas as pd
import pickle
import streamlit as st

# --------------------------- PAGE CONFIGURATION ---------------------------
st.set_page_config(
    page_title="💼 Salary Prediction App",
    page_icon="💰",
    layout="centered"
)

# --------------------------- CUSTOM CSS STYLING ---------------------------
st.markdown("""
    <style>
    body {
        background-color: #f5f7fa;
        font-family: 'Poppins', sans-serif;
    }
    .main {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 35px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin-top: 25px;
    }
    h1 {
        text-align: center;
        color: #0074D9;
        font-weight: 700;
        margin-bottom: 20px;
    }
    .stButton>button {
        background: linear-gradient(to right, #0074D9, #00BFFF);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 25px;
        font-size: 16px;
        font-weight: 600;
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #005fa3, #0099cc);
        transform: scale(1.05);
    }
    footer {
        text-align: center;
        color: #888;
        margin-top: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------------- LOAD MODEL & ENCODERS ---------------------------
# ✅ Load the model and encoders that were previously saved
with open("best_salary_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# --------------------------- APP TITLE ---------------------------
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.title("💼 Salary Prediction App")
st.markdown("### 🔍 Fill in the employee details below to predict their salary category:")

# --------------------------- SIDEBAR ---------------------------
st.sidebar.header("📘 About this App")
st.sidebar.write("""
### 💼 About the Salary Insight App

Gain data-backed insights into earning potential.  
This tool helps professionals and employers understand  
income ranges based on key personal and work attributes.  
""")

# --------------------------- FORM INPUTS ---------------------------
col1, col2 = st.columns(2)

with col1:
    workclass = st.selectbox("🏢 Workclass", encoders['workclass'].classes_, key="workclass")
    education = st.selectbox("🎓 Education", encoders['education'].classes_, key="education")
    education_no = st.number_input("🔢 Education Number", 1, 16, 10, key="education_no")
    occupation = st.selectbox("💼 Occupation", encoders['occupation'].classes_, key="occupation")
    sex = st.selectbox("⚧️ Sex", encoders['sex'].classes_, key="sex")
    maritalstatus = st.selectbox("💍 Marital Status", encoders['maritalstatus'].classes_, key="marital")

with col2:
    native = st.selectbox("🌍 Native Country", encoders['native'].classes_, key="native")
    age = st.number_input("🎂 Age", 17, 90, 30, key="age")
    hours_per_week = st.number_input("🕒 Hours per Week", 1, 100, 40, key="hours")
    capital_gain = st.number_input("💹 Capital Gain", 0, 100000, 0, key="gain")
    capital_loss = st.number_input("📉 Capital Loss", 0, 5000, 0, key="loss")
    race = st.selectbox("🏳️ Race", encoders['race'].classes_, key="race")
    relationship = st.selectbox("🤝 Relationship", encoders['relationship'].classes_, key="relation")

# --------------------------- PREDICTION ---------------------------
st.markdown("---")
if st.button("🔮 Predict Salary Category"):
    input_data = {
        'age': age,
        'workclass': encoders['workclass'].transform([workclass])[0],
        'education': encoders['education'].transform([education])[0],
        'educationno': education_no,
        'maritalstatus': encoders['maritalstatus'].transform([maritalstatus])[0],
        'occupation': encoders['occupation'].transform([occupation])[0],
        'relationship': encoders['relationship'].transform([relationship])[0],
        'race': encoders['race'].transform([race])[0],
        'sex': encoders['sex'].transform([sex])[0],
        'capitalgain': capital_gain,
        'capitalloss': capital_loss,
        'hoursperweek': hours_per_week,
        'native': encoders['native'].transform([native])[0]
    }

    df = pd.DataFrame([input_data])
    result = model.predict(df)

    if result[0] == ">50K":
        st.success("💰 **Predicted Salary Category: > 50K**")
        st.balloons()
        st.markdown("🎉 Excellent! This person belongs to a **high-income category.**")
    else:
        st.warning("📊 **Predicted Salary Category: ≤ 50K**")
        st.markdown("🚀 This person currently earns **≤ 50K**, with potential for growth.")

# --------------------------- FOOTER ---------------------------
st.markdown("---")
st.caption("🧠 Developed with passion using Streamlit, Python & Machine Learning 💻")
st.markdown("</div>", unsafe_allow_html=True)
