import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier  # Example model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load accident data (replace with your actual data loading)
def load_data():
    # Placeholder data - replace with your actual dataset loading
    data = {
        'DayOfWeek': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        'TimeOfDay': ['Morning', 'Afternoon', 'Evening', 'Night'],
        'Weather': ['Clear', 'Rain', 'Snow', 'Fog'],
        'RoadSurface': ['Dry', 'Wet', 'Ice', 'Snow'],
        'AccidentSeverity': ['Minor', 'Moderate', 'Severe']
    }
    df = pd.DataFrame([(d1, d2, d3, d4, d5)
                           for d1 in data['DayOfWeek']
                           for d2 in data['TimeOfDay']
                           for d3 in data['Weather']
                           for d4 in data['RoadSurface']
                           for d5 in data['AccidentSeverity']],
                      columns=['DayOfWeek', 'TimeOfDay', 'Weather', 'RoadSurface', 'AccidentSeverity'])

    #create a smaller dataset.
    df_sample = df.sample(n=500, random_state=42)

    # Encode categorical variables
    label_encoders = {}
    for col in ['DayOfWeek', 'TimeOfDay', 'Weather', 'RoadSurface']:
        le = LabelEncoder()
        df_sample[col] = le.fit_transform(df_sample[col])
        label_encoders[col] = le  # Store the encoders for later use

    # Prepare data for modeling
    X = df_sample.drop('AccidentSeverity', axis=1)
    y = df_sample['AccidentSeverity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_encoders

# Train a model (replace with your actual model loading/training)
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# Load data and train model
X_train, X_test, y_train, y_test, label_encoders = load_data()
model = train_model(X_train, y_train)

# Streamlit App
st.title("AI-Driven Accident Risk Assessment")

# User input using Streamlit widgets
day_of_week = st.selectbox("Day of Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
time_of_day = st.selectbox("Time of Day", ['Morning', 'Afternoon', 'Evening', 'Night'])
weather = st.selectbox("Weather", ['Clear', 'Rain', 'Snow', 'Fog'])
road_surface = st.selectbox("Road Surface", ['Dry', 'Wet', 'Ice', 'Snow'])

# Encode user inputs
input_data = pd.DataFrame({
    'DayOfWeek': [day_of_week],
    'TimeOfDay': [time_of_day],
    'Weather': [weather],
    'RoadSurface': [road_surface],
})
for col in ['DayOfWeek', 'TimeOfDay', 'Weather', 'RoadSurface']:
    input_data[col] = label_encoders[col].transform(input_data[col])

# Make prediction
if st.button("Predict Risk"):
    prediction = model.predict(input_data)
    # Get probabilities for each class
    probabilities = model.predict_proba(input_data)[0]
    #st.write(probabilities)
    # Display prediction
    st.subheader("Predicted Accident Severity:")
    st.write(prediction[0])

    # Display probabilities (optional, for more detailed output)
    st.subheader("Probabilities:")
    for i, severity_level in enumerate(model.classes_):
        st.write(f"{severity_level}: {probabilities[i]:.2f}")
