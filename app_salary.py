import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

### Load the trained model, scaler pickle, onehot
model=tf.keras.models.load_model('artifacts/model_salary.h5')

## load the encoder and scaler
with open('artifacts/onehot_encoder_geo.pkl','rb') as file: # rb = read bytes
    onehot_encoder_geo = pickle.load(file)

with open('artifacts/onehot_encoder_gender.pkl', 'rb') as file:
    onehot_encoder_gender = pickle.load(file)

with open('artifacts/scaler_salary.pkl', 'rb') as file:
    scaler = pickle.load(file)


## streamlit app
st.title('Customer Salary Prediction: ')

# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', onehot_encoder_gender.categories_[0])
age = st.slider('Age', 18, 100)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
churn = st.selectbox('Churn', [0, 1])
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [churn]
})

# One-hot encode 'Gender' and 'Geography'
gender_encoded = onehot_encoder_gender.transform([[gender]]).toarray()
gender_encoded_df = pd.DataFrame(gender_encoded, columns=onehot_encoder_gender.get_feature_names_out(['Gender']))

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns
encoded_df = pd.concat([gender_encoded_df, geo_encoded_df], axis=1)

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Salary Prediction: {prediction_proba:.2f}')