import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load and preprocess the wine quality dataset
wine_dataset = pd.read_csv("winequality-red.csv")


def preprocess_data(data):
    X = data.drop("quality", axis=1)
    Y = data["quality"].apply(lambda y_value: 1 if y_value >= 7 else 0)
    return X, Y


X, Y = preprocess_data(wine_dataset.copy())

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Train the Random Forest Classifier model
model = RandomForestClassifier()
model.fit(X_train, Y_train)


# Function to make predictions
def predict_wine_quality(data):
    data_array = np.asarray(data).reshape(1, -1)
    prediction = model.predict(data_array)[0]
    if prediction == 1:
        return "Good Quality Wine"
    else:
        return "Bad Quality Wine"


# Streamlit frontend
st.title("Wine Quality Prediction")

st.write("Enter the following wine characteristics to predict its quality:")

# Input fields for wine characteristics
fixed_acidity = st.number_input(
    "Fixed Acidity", min_value=0.0, max_value=16.0, step=0.01
)
volatile_acidity = st.number_input(
    "Volatile Acidity", min_value=0.0, max_value=1.58, step=0.01
)
citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=1.00, step=0.01)
residual_sugar = st.number_input(
    "Residual Sugar", min_value=0.0, max_value=45.0, step=0.1
)
chlorides = st.number_input("Chlorides", min_value=0.0, max_value=2.00, step=0.01)
free_sulfur_dioxide = st.number_input(
    "Free Sulfur Dioxide", min_value=0.0, max_value=200.0, step=1.0
)
total_sulfur_dioxide = st.number_input(
    "Total Sulfur Dioxide", min_value=0.0, max_value=400.0, step=1.0
)
density = st.number_input("Density", min_value=0.980, max_value=1.038, step=0.001)
pH = st.number_input("pH", min_value=3.0, max_value=4.0, step=0.01)
sulphates = st.number_input("Sulphates", min_value=0.0, max_value=2.00, step=0.01)
alcohol = st.number_input("Alcohol", min_value=8.0, max_value=14.0, step=0.1)

# User input data
user_data = [
    fixed_acidity,
    volatile_acidity,
    citric_acid,
    residual_sugar,
    chlorides,
    free_sulfur_dioxide,
    total_sulfur_dioxide,
    density,
    pH,
    sulphates,
    alcohol,
]

# Predict button
if st.button("Predict Wine Quality"):
    prediction = predict_wine_quality(user_data)
    st.success(f"The wine is predicted to be of {prediction}.")

st.write("---")
