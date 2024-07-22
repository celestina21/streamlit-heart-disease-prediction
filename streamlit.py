import streamlit as st
import pandas as pd
import pickle

# Load the trained model
model = pickle.load(open('trained_heart_disease_classifier_model.pkl', 'rb'))

# Define a function to make predictions
def predict_heart_disease(age, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    data = {
        'age': [age],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    }
    df = pd.DataFrame(data)
    prediction = model.predict(df)
    return prediction[0]

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="wide",
    page_icon="❤️"
)

with st.container():
    # Center the title
    col1, col2, col3 = st.columns([2, 4, 0.5])
    with col2:
        st.title("Heart Disease Prediction")
st.write("Provide patient details to predict the likelihood of heart disease.")

# Sidebar for inputs
with st.sidebar:
    st.header("Patient Details")
    age = st.slider("Age", min_value=0, max_value=120, value=30)
    cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", min_value=50, max_value=200, value=120)
    chol = st.slider("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False", "True"])
    restecg = st.selectbox("Resting Electrocardiographic Results", ["normal", "stt abnormality", "lv hypertrophy"])
    thalach = st.slider("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
    exang = st.selectbox("Exercise-Induced Angina", ["False", "True"])
    oldpeak = st.slider("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", ["downsloping", "flat", "upsloping"])
    ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=3, value=0)
    thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversible defect"])

# Convert categorical inputs
cp_mapping = {"typical angina": 1, "atypical angina": 2, "non-anginal": 3, "asymptomatic": 4}
fbs_mapping = {"False": 0, "True": 1}
restecg_mapping = {"normal": 0, "stt abnormality": 1, "lv hypertrophy": 2}
exang_mapping = {"False": 0, "True": 1}
slope_mapping = {"downsloping": 0, "flat": 1, "upsloping": 2}
thal_mapping = {"normal": 0, "fixed defect": 1, "reversible defect": 2}

cp = cp_mapping[cp]
fbs = fbs_mapping[fbs]
restecg = restecg_mapping[restecg]
exang = exang_mapping[exang]
slope = slope_mapping[slope]
thal = thal_mapping[thal]

# Prediction and result display
if st.button("Predict", key='predict_button'):
    result = predict_heart_disease(age, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    if result == 1:
        st.success("The model predicts that the patient has heart disease.", icon="✅")
    else:
        st.success("The model predicts that the patient does not have heart disease.", icon="✅")

# Add some spacing
st.write("\n")
st.write("\n")
