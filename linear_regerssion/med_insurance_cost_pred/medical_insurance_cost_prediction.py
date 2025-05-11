# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# """Data Collection & Analysis"""

# loading the data from csv file to a Pandas DataFrame
insurance_dataset = pd.read_csv('./insurance.csv')

# encoding sex column
insurance_dataset.replace({'sex':{'male':0,'female':1}}, inplace=True)
3 # encoding 'smoker' column
insurance_dataset.replace({'smoker':{'yes':0,'no':1}}, inplace=True)
# encoding 'region' column
insurance_dataset.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}}, inplace=True)

# """Splitting the Features and Target"""

X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']
# """Splitting the data into Training data & Testing Data"""
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

# loading the Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
training_data_prediction =regressor.predict(X_train)
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R squared vale : ', r2_train)
# prediction on test data
test_data_prediction =regressor.predict(X_test)
# R squared value
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R squared vale : ', r2_test)


# prompt: # changing input_data to a numpy array
# input_data_as_numpy_array = np.asarray(input_data)
# # reshape the array
# input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
# prediction = regressor.predict(input_data_reshaped)
# print(prediction)
# print('The insurance cost is USD ', prediction[0])
# Convert above code if I use streamlit

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load the model and data (replace with your actual file paths)
insurance_dataset = pd.read_csv('/content/insurance.csv')  # Make sure this path is correct for your Streamlit environment

# Preprocessing (same as your original code)
insurance_dataset.replace({'sex':{'male':0,'female':1}}, inplace=True)
insurance_dataset.replace({'smoker':{'yes':0,'no':1}}, inplace=True)
insurance_dataset.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}}, inplace=True)

X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


# Streamlit app
st.title('Insurance Cost Prediction')

# Input fields
age = st.number_input('Age', min_value=18, max_value=100, value=31)
sex = st.selectbox('Sex', ['Male', 'Female'])
bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=25.74)
children = st.number_input('Children', min_value=0, max_value=5, value=0)
smoker = st.selectbox('Smoker', ['Yes', 'No'])
region = st.selectbox('Region', ['Southeast', 'Southwest', 'Northeast', 'Northwest'])


# Preprocess user input
input_data = [age, 1 if sex == 'Female' else 0, bmi, children, 1 if smoker == 'No' else 0,
              0 if region == 'Southeast' else (1 if region == 'Southwest' else (2 if region == 'Northeast' else 3))]

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# Make prediction
prediction = regressor.predict(input_data_reshaped)

# Display prediction
st.write(f'The insurance cost is USD {prediction[0]:.2f}')