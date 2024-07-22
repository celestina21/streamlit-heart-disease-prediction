import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler, OneHotEncoder

# Load the trained model
model = joblib.load('trained_heart_disease_classifier_model.pkl')

# Function to take in user inputs, store it as a DataFrame, then use the DataFrame to predict whether or not the patient has heart disease 
def predict_heart_disease(age, chest_pain_type, top_rest_bps, cholesterol, high_fasting_blood_sugar, restecg, max_heart_rate, exercise_induced_angina, st_depression, slope, colored_vessels, thalassemia):
    data = {
        'age': [age],
        'chest_pain_type': [chest_pain_type],
        'top_rest_bps': [top_rest_bps],
        'cholesterol': [cholesterol],
        'high_fasting_blood_sugar': [high_fasting_blood_sugar],
        'restecg': [restecg],
        'max_heart_rate': [max_heart_rate],
        'exercise_induced_angina': [exercise_induced_angina],
        'st_depression': [st_depression],
        'slope': [slope],
        'colored_vessels': [colored_vessels],
        'thalassemia': [thalassemia]
    }
    df = pd.DataFrame(data)
    prediction = model.predict(df)
    return prediction[0]

# Set page configuration
st.set_page_config(
    page_title='Heart Disease Prediction',
    layout='wide',
    page_icon='❤️'
)

with st.container():
    # Center the title
    col1, col2, col3 = st.columns([2, 4, 0.5])
    with col2:
        st.title('Heart Disease Prediction')
st.write('Provide patient details to predict the likelihood of heart disease.')

# Sidebar for inputs
with st.sidebar:
    st.header('Patient Details')
    age = st.slider('Age', min_value = 0, max_value = 100, value = 50)
    chest_pain_type = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-Anginal', 'Asymptomatic'])
    top_rest_bps = st.slider('Systolic Resting Blood Pressure (mm Hg)', min_value = 80, max_value = 200, value = 120)
    cholesterol = st.slider('Serum Cholesterol (mg/dl)', min_value = 100, max_value = 600, value = 200)
    high_fasting_blood_sugar = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['False', 'True'])
    restecg = st.selectbox('Resting Electrocardiographic Results', ['Normal', 'ST-T Abnormality', 'LV Hypertrophy'])
    max_heart_rate = st.slider('Maximum Heart Rate Achieved During Stress Test (bpm)', min_value = 50, max_value = 250, value = 150)
    exercise_induced_angina = st.selectbox('Has Exercise-Induced Angina', ['False', 'True'])
    st_depression = st.slider('ST Depression Induced by Exercise (mm)', min_value = 0.0, max_value = 10.0, value = 1.0)
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', ['Downsloping', 'Flat', 'Upsloping'])
    colored_vessels = st.slider('Number of Major Vessels Colored by Fluoroscopy', min_value = 0, max_value = 3, value = 0)
    thalassemia = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])

# Prediction and result display
if st.button('Predict', key = 'predict_button'):
    result = predict_heart_disease(age, chest_pain_type.lower(), top_rest_bps, cholesterol, high_fasting_blood_sugar, restecg.lower(), max_heart_rate, exercise_induced_angina, st_depression, slope.lower(), colored_vessels, thalassemia.lower())
    if result == 1:
        st.success('The model predicts that the patient has heart disease.', icon='✅')
    else:
        st.success('The model predicts that the patient does not have heart disease.', icon='❎')

# Add some spacing
st.write('\n')
st.write('\n')
