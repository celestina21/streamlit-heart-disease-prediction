import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler, OneHotEncoder

# Load the best model
model = joblib.load('trained_heart_disease_classifier_model.pkl')

# Define a data dictionary 
data_dict = {
    'Feature': [
        'Age',
        'Chest Pain Type',
        'Systolic Resting Blood Pressure (mm Hg)',
        'Serum Cholesterol (mg/dl)',
        'Fasting Blood Sugar > 120 mg/dl',
        'Resting ECG Results',
        'Maximum Heart Rate (bpm)',
        'Exercise-Induced Angina',
        'ST Depression Induced (mm)',
        'Slope',
        'Number of Major Vessels Colored',
        'Thalassemia'
    ],
    'Description': [
        "Patient's age in years",
        'Type of chest pain experienced by the patient',
        'Systolic/Top blood pressure in mm Hg at rest',
        'Serum cholesterol level in mg/dl',
        "True if patient's fasting blood sugar is above 120 mg/dl",
        'Results from resting electrocardiographic test',
        'Maximum heart rate achieved during stress test',
        'True if patient has exercise-induced angina',
        'ST depression induced by exercise relative to rest',
        'Slope of the peak exercise ST segment',
        'Number of major vessels colored by fluoroscopy',
        'Thalassemia type'
    ]
}
# Convert the data dictionary to DataFrame
df_data_dict = pd.DataFrame(data_dict)

# Define result as 2 so that no predictions are displayed when page is first opened or is refreshed 
result = 2

# Function to take in user inputs, store it as a DataFrame, then use the DataFrame to predict whether or not the patient has heart disease
def predict_heart_disease(age, chest_pain_type, top_rest_bps, cholesterol, high_fasting_blood_sugar, restecg, max_heart_rate, exercise_induced_angina, st_depression, slope, colored_vessels, thalassemia):
    '''
    Takes in the user's inputs from the sidebar.
    Returns the prediction of whether the user has (1) or does not have (0) heart disease.
    '''
    data = {
        'age': age,
        'chest_pain_type': chest_pain_type,
        'top_rest_bps': top_rest_bps,
        'cholesterol': cholesterol,
        'high_fasting_blood_sugar': high_fasting_blood_sugar,
        'restecg': restecg,
        'max_heart_rate': max_heart_rate,
        'exercise_induced_angina': exercise_induced_angina,
        'st_depression': st_depression,
        'slope': slope,
        'colored_vessels': colored_vessels,
        'thalassemia': thalassemia
    }
    df = pd.DataFrame(data, index = [0])
    # Since the model is a pipeline, the one-hot encoding and robust scaling of the inputted data is handled at this step as well 
    prediction = model.predict(df)
    return prediction[0]


# Change the page title and favicon as well as use a wide layput so that elements take up all available space 
st.set_page_config(
    page_title = 'Heart Disease Prediction',
    layout = 'wide',
    page_icon = '❤️'
)

# Use a container with column layout that centers the website title both when the sidebar is expanded and collapsed 
with st.container():
    col1, col2, col3 = st.columns([1.5, 3, 0.5])
    with col2:
        st.title('Heart Disease Prediction')

# Use a container with column layout that centers instructions for users both when the sidebar is expanded and collapsed 
with st.container():
    col1, col2, col3 = st.columns([2, 3, 1.5])
    with col2:
        st.markdown('#### 1. Open the left sidebar.')
        st.markdown('#### 2. Provide patient details.')

# Add some spacing
st.write('\n')
        
# Sidebar with inputs. All inputs have value or placeholder specified to display the default value used if the user adds nothing 
# slides and selectboxes are used to ensure only acceptable values will be used for prediction
with st.sidebar:
    st.markdown('# Patient Details')
    age = st.slider('Age', min_value = 0, max_value = 120, value = 50)
    chest_pain_type = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-Anginal', 'Asymptomatic'], placeholder = 'Typical Angina')
    top_rest_bps = st.slider('Systolic Resting Blood Pressure (mm Hg)', min_value = 80, max_value = 200, value = 120)
    cholesterol = st.slider('Serum Cholesterol (mg/dl)', min_value = 100, max_value = 600, value = 200)
    high_fasting_blood_sugar = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['False', 'True'], placeholder = 'False')
    restecg = st.selectbox('Resting ECG Results', ['Normal', 'ST-T Abnormality', 'LV Hypertrophy'], placeholder = 'Normal')
    max_heart_rate = st.slider('Maximum Heart Rate (bpm)', min_value = 50, max_value = 250, value = 150)
    exercise_induced_angina = st.selectbox('Exercise-Induced Angina', ['False', 'True'], placeholder = 'False')
    st_depression = st.slider('ST Depression (mm)', min_value = 0.0, max_value = 10.0, value = 1.0)
    slope = st.selectbox('Slope', ['Downsloping', 'Flat', 'Upsloping'], placeholder = 'Downsloping')
    colored_vessels = st.slider('Number of Major Vessels Colored', min_value = 0, max_value = 3, value = 0)
    thalassemia = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'], placeholder = 'Normal')

# Use a container with column layout that centers Predict button both when the sidebar is expanded and collapsed 
with st.container():
    col1, col2, col3 = st.columns([2.3, 3, 2.5])
    with col2:
        # Predict button to pass user's inputs into predict_heart_disease() and store the prediction as result. 
        if st.button('Predict', key = 'predict_button', type = 'primary', use_container_width = True):
            result = predict_heart_disease(age, chest_pain_type.lower(), top_rest_bps, cholesterol, high_fasting_blood_sugar, restecg.lower(), max_heart_rate, exercise_induced_angina, st_depression, slope.lower(), colored_vessels, thalassemia.lower())

# When result is 0 or 1, meaning a prediction by the model was made, display the prediction in a readable form
if result == 1:
    st.success('This patient likely has heart disease.', icon = '✅')
    st.info('However, exercise caution and verify this result before completing diagnosis.', icon = '⚠️')
elif result == 0:
    st.success('This patient likely does not have heart disease.', icon = '❎')
    st.info('Exercise caution and verify this result before completing diagnosis.', icon = '⚠️')
# If result remains as 2, no prediction was made so no output is shown yet 
else: 
    pass

# Add some spacing
st.write('\n')
st.write('\n')

# Use a container with column layout that aligns a display of the user's inputs and the data dictionary horixontally and centers both when sidebar is expanded and collapsed
with st.container():
    col1, col2, col3, col4 = st.columns([3, 5, 1, 9])
    with col2:
        # Display user inputs 
        st.markdown('## Inputs')
        st.write(f'**Age**: {age}')
        st.write(f'**Chest Pain Type**: {chest_pain_type}')
        st.write(f'**Systolic Resting Blood Pressure (mm Hg)**: {top_rest_bps}')
        st.write(f'**Serum Cholesterol (mg/dl)**: {cholesterol}')
        st.write(f'**Fasting Blood Sugar > 120 mg/dl**: {high_fasting_blood_sugar}')
        st.write(f'**Resting ECG Results**: {restecg}')
        st.write(f'**Maximum Heart Rate (bpm)**: {max_heart_rate}')
        st.write(f'**Exercise-Induced Angina**: {exercise_induced_angina}')
        st.write(f'**ST Depression (mm)**: {st_depression}')
        st.write(f'**Slope**: {slope}')
        st.write(f'**Number of Major Vessels Colored**: {colored_vessels}')
        st.write(f'**Thalassemia**: {thalassemia}')
    with col4:
        st.markdown('## Data Dictionary')
        st.dataframe(df_data_dict, hide_index = True, height = 455)