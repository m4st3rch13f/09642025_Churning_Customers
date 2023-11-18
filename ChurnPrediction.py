import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pandas as pd
from scipy.stats import norm

# Load the model
model = load_model('churn_model.h5')

# Feature names
features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'gender', 'Partner',
            'Dependents', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod', 'SeniorCitizen']

# Load the StandardScaler
sc = StandardScaler()

# Load the LabelEncoder for categorical features
le = LabelEncoder()

# Function to preprocess user input
def preprocess_input(user_input):
    # Perform any necessary preprocessing on the user input
    # For example, scale numerical features and encode categorical features

    # Scale numerical features
    scaled_input = sc.fit_transform(user_input[['tenure', 'MonthlyCharges', 'TotalCharges']])

    # Encode categorical features
    encoded_input = user_input.copy()
    for feature in ['gender', 'Partner', 'Dependents', 'MultipleLines',
                    'InternetService', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                    'Contract', 'PaperlessBilling', 'PaymentMethod',
                    'SeniorCitizen']:
        encoded_input[feature] = le.fit_transform(user_input[feature])

    # Combine the scaled and encoded features
    preprocessed_input = np.concatenate([scaled_input, encoded_input], axis=1)

    return preprocessed_input

# Calculate the confidence interval
def calculate_confidence_interval(prediction, std_error, confidence_level=0.95):
    z_score = norm.ppf((1 + confidence_level) / 2)
    margin_of_error = z_score * std_error
    lower_bound = prediction - margin_of_error
    upper_bound = prediction + margin_of_error
    return lower_bound, upper_bound

# Streamlit app
def main():
    st.title('Customer Churn Prediction')

    # Create input elements for each feature
    tenure = st.slider('Tenure (months)', min_value=1, max_value=72, value=1)
    monthly_charges = st.slider('Monthly Charges', min_value=0.0, max_value=1000.0, value=0.0)
    total_charges = st.slider('Total Charges', min_value=0.0, max_value=10000.0, value=0.0)

    # Categorical features
    gender = st.selectbox('Gender', ['Male', 'Female'])
    partner = st.selectbox('Partner', ['Yes', 'No'])
    dependents = st.selectbox('Dependents', ['Yes', 'No'])
    multiple_lines = st.selectbox('Multiple Lines', ['No phone service', 'No', 'Yes'])
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
    online_backup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
    device_protection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
    tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
    streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
    streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
    payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])

    # Create a DataFrame with the user input
    user_input = pd.DataFrame({'tenure': [tenure],
                               'MonthlyCharges': [monthly_charges],
                               'TotalCharges': [total_charges],
                               'gender': [gender],
                               'Partner': [partner],
                               'Dependents': [dependents],
                               'MultipleLines': [multiple_lines],
                               'InternetService': [internet_service],
                               'OnlineSecurity': [online_security],
                               'OnlineBackup': [online_backup],
                               'DeviceProtection': [device_protection],
                               'TechSupport': [tech_support],
                               'StreamingTV': [streaming_tv],
                               'StreamingMovies': [streaming_movies],
                               'Contract': [contract],
                               'PaperlessBilling': [paperless_billing],
                               'PaymentMethod': [payment_method],
                               'SeniorCitizen': [senior_citizen]})

    # Preprocess the user input
    processed_input = preprocess_input(user_input)

    # Predict button
    if st.button('Predict'):
        # Make prediction
        prediction = model.predict(processed_input)[0]

        # Calculate the confidence interval (replace 0.1 with the actual standard error)
        std_error = 0.1
        confidence_interval = calculate_confidence_interval(prediction, std_error)

        # Display the prediction result and confidence interval
        if prediction > 0.5:
            st.warning(f'The customer is predicted to churn with a confidence interval of {confidence_interval}.')
        else:
            st.success(f'The customer is predicted not to churn with a confidence interval of {confidence_interval}.')

    # Reset button
    if st.button('Reset'):
        st.experimental_rerun()

# Run the app
if __name__ == '__main__':
    main()
