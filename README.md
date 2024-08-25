# Streamlit Heart Disease Prediction Website 
## A streamlit website that integrated a binary classifier to predict if a patient is at risk of heart disease based on various medical data 
[View website here](https://app-heart-disease-prediction-lhu5e6yvqydqha4p2vmnh9.streamlit.app/)
### Created for Machine Learning for Developers module

Libraries used: 
- Pandas
- Scikit-learn
- Scipy
- Numpy
- Matplotlib
- Seaborn 
- Joblib


Files:
- heart_disease_uci.csv: Dataset retrieved from Kaggle. ([Source](https://www.kaggle.com/datasets/haiderrasoolqadri/heart-disease-dataset-uci))
- Project.ipynb: Jupyter notebook wherein data exploration and cleaning, model training of various classifiers, hyper-parameter tuning, model selection and export of best performing model occurred.  
- trained_heart_disease_classifier_model.pkl: Exported model. (Support vector classifier with an accuracy of 0.83. Other metrics can be viewed in Project.ipynb or Report.docx sections "Results After Hyper-Parameter Tuning" and "Results and Analysis")
- streamlit.py: Streamlit code to create website interface and integrate exported model for predictions.
- .streamlit/config.toml: Custom configurations for design of Streamlit website.
- requirements.txt: Project dependencies for deployment of project using Streamlit.
- Report.docx: A project report detailing and justifying data exploration and cleaning done, as well as evaluating performance of classifiers used to select the best performing model.
