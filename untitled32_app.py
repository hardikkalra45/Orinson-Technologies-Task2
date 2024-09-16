import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier

# Load the saved model
model = pickle.load(open('C:/Users/hardi/Downloads/model.pkl', 'rb'))

# Title of the app
st.title('ML Model Deployment with Streamlit')

# Instructions
st.write("""
This is a simple web application to predict using a machine learning model.
""")

# Input features
def user_input_features():
    feature1 = st.number_input("Enter value for feature 1", 0.0, 8.0, 5.0)
    feature2 = st.number_input("Enter value for feature 2", 0.0, 4.0, 2.0)
    feature3 = st.number_input("Enter value for feature 3", 0.0, 7.0, 0.0)
    feature4 = st.number_input("Enter value for feature 4", 0.0, 3.0, 0.5)
    data = {
        'sepal length (cm)': feature1,
        'sepal width (cm)': feature2,
        'petal length (cm)': feature3,
        'petal width (cm)': feature4
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Store user input
input_df = user_input_features()

# Display user input
st.subheader('User Input:')
st.write(input_df)

# Make predictions
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Display the results
st.subheader('Prediction:')
st.write('Class: ', prediction)
st.subheader('Prediction Probability:')
st.write(prediction_proba)
