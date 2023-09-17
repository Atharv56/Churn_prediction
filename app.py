import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier

model_file_path = os.path.abspath('churn_model.pkl')

with open(model_file_path, 'rb') as file:
    model = pickle.load(file)

st.title('Churn Prediction Web App')
st.sidebar.header('User Input Features')

age = st.sidebar.slider('Age', 18, 100, 30)
subscription_length = st.sidebar.slider('Subscription Length (Months)', 1, 60, 12)
monthly_bill = st.sidebar.slider('Monthly Bill', 0.0, 500.0, 50.0)
total_usage_gb = st.sidebar.slider('Total Usage (GB)', 0, 1000, 100)
gender = st.sidebar.radio('Gender', ['Male', 'Female'])
is_location_houston = st.sidebar.checkbox('Location Houston')
is_location_los_angeles = st.sidebar.checkbox('Location Los Angeles')
is_location_miami = st.sidebar.checkbox('Location Miami')
is_location_new_york = st.sidebar.checkbox('Location New York')

gender_mapping = {'Male': 1, 'Female': 0}
gender_encoded = gender_mapping[gender]

user_input = pd.DataFrame({
    'Age': [age],
    'Gender': [gender_encoded],
    'Subscription_Length_Months': [subscription_length],
    'Monthly_Bill': [monthly_bill],
    'Total_Usage_GB': [total_usage_gb],
    'Location_Houston': [is_location_houston],
    'Location_Los Angeles': [is_location_los_angeles],
    'Location_Miami': [is_location_miami],
    'Location_New York': [is_location_new_york]
})
from sklearn.preprocessing import MinMaxScaler
columns_to_normalize = ['Age', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB']
scaler = MinMaxScaler()
user_input[columns_to_normalize] = scaler.fit_transform(user_input[columns_to_normalize])

def predict_churn(user_input):
    prediction = model.predict(user_input)[0]
    return prediction


if st.button('Predict'):
    result = predict_churn(user_input)
    if result == 1:
        st.write('Churn Prediction: Churn')
    else:
        st.write('Churn Prediction: No Churn')
