import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import svm

# Load Parkinson's disease dataset
parkinsons_data = pd.read_csv('parkinsons.csv')

# Remove non-numeric columns if any
parkinsons_data = parkinsons_data.select_dtypes(include=[np.number])

# Train-test split for Parkinson's disease data
X_parkinsons = parkinsons_data.drop(columns=['status'])
Y_parkinsons = parkinsons_data['status']
X_train_parkinsons, X_test_parkinsons, Y_train_parkinsons, Y_test_parkinsons = train_test_split(X_parkinsons, Y_parkinsons, test_size=0.2, random_state=2)

# Train SVM model for Parkinson's disease prediction
scaler_parkinsons = StandardScaler()
X_train_scaled_parkinsons = scaler_parkinsons.fit_transform(X_train_parkinsons)
X_test_scaled_parkinsons = scaler_parkinsons.transform(X_test_parkinsons)

parkinsons_model = svm.SVC(kernel='linear')
parkinsons_model.fit(X_train_scaled_parkinsons, Y_train_parkinsons)

# Function to predict Parkinson's disease
def predict_parkinsons(input_data):
    scaled_input_data = scaler_parkinsons.transform(input_data)
    prediction = parkinsons_model.predict(scaled_input_data)
    return prediction

# Your remaining code...


# Load heart disease dataset
heart_data = pd.read_csv('heart.csv')

# Split data for heart disease model
X_heart = heart_data.drop(columns=['target'], axis=1)
Y_heart = heart_data['target']

# Train-test split for heart disease data
X_train_heart, X_test_heart, Y_train_heart, Y_test_heart = train_test_split(X_heart, Y_heart, test_size=0.2, random_state=2)

# Train SVM model for heart disease prediction
scaler_heart = StandardScaler()
X_train_scaled_heart = scaler_heart.fit_transform(X_train_heart)
X_test_scaled_heart = scaler_heart.transform(X_test_heart)

heart_model = svm.SVC(kernel='linear')
heart_model.fit(X_train_scaled_heart, Y_train_heart)

# Function to predict heart disease
def predict_heart_disease(input_data):
    scaled_input_data_heart = scaler_heart.transform(input_data)
    prediction_heart = heart_model.predict(scaled_input_data_heart)
    return prediction_heart

# Display header
st.title('Disease Prediction')

# Checkbox to choose disease
disease = st.checkbox("Parkinson's Disease")

# Function to input data using sliders
def input_data_ui(features):
    input_data = []
    for i, feature in enumerate(features):
        val = st.slider(feature, min_value=0.0, max_value=100.0, step=0.1)
        input_data.append(val)
    return np.asarray(input_data).reshape(1, -1)

# Get column names for Parkinson's disease features
parkinsons_features = X_parkinsons.columns.tolist()

# Get input data
if disease:
    input_data = input_data_ui(parkinsons_features)
else:
    input_data = input_data_ui(X_heart.columns[:-1])  # Exclude the target column

# Button to apply entered data
if st.button('Apply') and input_data is not None:
    # Perform prediction based on selected disease
    if disease:
        prediction = predict_parkinsons(input_data)
        if prediction[0] == 0:
            st.write("The Person does not have Parkinson's Disease")
        else:
            st.write("The Person has Parkinson's Disease")
    else:
        prediction = predict_heart_disease(input_data)
        if prediction[0] == 0:
            st.write("The Person does not have Heart Disease")
        else:
            st.write("The Person has Heart Disease")



# Define function to load and preprocess data
def load_data():
    diabetes_dataset = pd.read_csv('diabetes.csv')
    X = diabetes_dataset.drop(columns='Outcome', axis=1)
    Y = diabetes_dataset['Outcome']
    scaler = StandardScaler()
    scaler.fit(X)
    X_standardized = scaler.transform(X)
    return X_standardized, Y, scaler

# Define function to train SVM classifier
def train_model(X_train, Y_train):
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, Y_train)
    return classifier

# Define function to make predictions
def predict_data(classifier, scaler, input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    std_data = scaler.transform(input_data_reshaped)
    prediction = classifier.predict(std_data)
    return prediction

# Define Streamlit app
def main():
    st.title('Diabetes Prediction')

    # Load and preprocess data
    X, Y, scaler = load_data()

    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    # Train model
    classifier = train_model(X_train, Y_train)

    # Display accuracy scores
    st.write('Training Data Accuracy:', accuracy_score(classifier.predict(X_train), Y_train))
    st.write('Test Data Accuracy:', accuracy_score(classifier.predict(X_test), Y_test))

    # Input data for prediction
    input_data = st.text_input('Enter input data separated by comma (e.g., 5,166,72,19,175,25.8,0.587,51):')
    input_data = [float(x.strip()) for x in input_data.split(',')] if input_data else None

    if input_data:
        # Make prediction
        prediction = predict_data(classifier, scaler, input_data)

        # Display prediction result
        if prediction[0] == 0:
            st.write('The person is not diabetic')
        else:
            st.write('The person is diabetic')

if __name__ == '__main__':
    main()

