import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pandas as pd
from scipy.stats import norm

# Load the model
model = load_model('churn_final_model.h5')

# Feature names
features = ['tenure', 'PaperlessBilling', 'Contract', 'StreamingMovies',
       'StreamingTV', 'TechSupport', 'DeviceProtection', 'OnlineBackup',
       'PaymentMethod', 'OnlineSecurity']

# Load the StandardScaler
sc = StandardScaler()

# Load the LabelEncoder for categorical features
le = LabelEncoder()

# Function to preprocess user input
def preprocess_input(user_input):
    # Perform any necessary preprocessing on the user input
    # For example, scale numerical features and encode categorical features

    # Scale numerical features
    scaled_input = sc.fit_transform(user_input[['tenure']])

    # Encode categorical features
    encoded_input = user_input.copy()
    for feature in['PaperlessBilling', 'Contract', 'StreamingMovies',
                    'StreamingTV', 'TechSupport', 'DeviceProtection', 'OnlineBackup',
                    'PaymentMethod', 'OnlineSecurity']:
        encoded_input[feature] = le.fit_transform(user_input[feature])

    # Combine the scaled and encoded features
    preprocessed_input = np.concatenate([scaled_input, encoded_input.drop('tenure', axis=1)], axis=1)
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

    # Categorical features
    online_backup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
    online_security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
    device_protection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
    tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
    streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
    streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
    payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

    # Create a DataFrame with the user input
    user_input = pd.DataFrame({'tenure': [tenure],
                               'OnlineSecurity': [online_security],
                               'OnlineBackup': [online_backup],
                               'DeviceProtection': [device_protection],
                               'TechSupport': [tech_support],
                               'StreamingTV': [streaming_tv],
                               'StreamingMovies': [streaming_movies],
                               'Contract': [contract],
                               'PaperlessBilling': [paperless_billing],
                               'PaymentMethod': [payment_method]})

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
        confidence_level = 0.95  # Set the desired confidence level
        if prediction > 0.5:
            lower_bound, upper_bound = calculate_confidence_interval(prediction, std_error, confidence_level)
            st.warning(f'The customer is predicted to churn with a {confidence_level * 100:.0f}% confidence interval of ({lower_bound[0]:.4f}, {upper_bound[0]:.4f}).')
        else:
            lower_bound, upper_bound = calculate_confidence_interval(prediction, std_error, confidence_level)
            st.success(f'The customer is predicted not to churn with a {confidence_level * 100:.0f}% confidence interval of ({lower_bound[0]:.4f}, {upper_bound[0]:.4f}).')

    # Reset button
    if st.button('Reset'):
        st.session_state.tenure = 1
        st.session_state.online_backup = 'No'
        # ... repeat for other features
        st.experimental_rerun()

# Run the app
if __name__ == '__main__':
    main()
