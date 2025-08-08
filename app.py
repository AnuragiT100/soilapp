<<<<<<< HEAD
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Sample data and model setup (same as in your training)
soil_data = pd.DataFrame({
    'Gravel': [24.1, 9.8, 14.0, 7.5, 30.0, 15.0, 40.0],
    'Sand': [74.8, 89.1, 85.4, 91.6, 65.0, 80.0, 55.0],
    'Fines': [1.2, 1.0, 0.6, 0.9, 5.0, 4.5, 5.0],
    'D10': [0.22, 0.30, 0.20, 0.20, 0.15, 0.25, 0.10],
    'D30': [0.48, 0.50, 0.32, 0.40, 0.35, 0.45, 0.25],
    'D50': [0.90, 0.70, 0.60, 0.60, 0.50, 0.60, 0.40],
    'D60': [2.00, 1.00, 0.8, 0.90, 1.50, 1.80, 0.70],
    'Cu': [9.09, 3.33, 4.00, 4.50, 10.0, 7.2, 7.0],
    'Cc': [0.52, 0.83, 0.64, 0.89, 1.0, 1.1, 1.2],
    'Gs': [2.5, 2.7, 2.5, 2.5, 2.6, 2.65, 2.55],
    'USCS': ['SW', 'SW', 'SW', 'SW', 'SP', 'SP', 'GP']
})

label_encoder = LabelEncoder()
soil_data['label'] = label_encoder.fit_transform(soil_data['USCS'])

X = soil_data.drop(['USCS', 'label'], axis=1)
y = soil_data['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Build and train the model (ideally, you save and load model later)
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_scaled, y, epochs=100, verbose=0)

# Streamlit UI
st.title("Soil Classification Predictor")

def user_input():
    gravel = st.number_input("Gravel (%)", min_value=0.0, max_value=100.0, value=20.0)
    sand = st.number_input("Sand (%)", min_value=0.0, max_value=100.0, value=75.0)
    fines = st.number_input("Fines (%)", min_value=0.0, max_value=100.0, value=5.0)
    D10 = st.number_input("D10 (mm)", value=0.25)
    D30 = st.number_input("D30 (mm)", value=0.35)
    D50 = st.number_input("D50 (mm)", value=0.50)
    D60 = st.number_input("D60 (mm)", value=1.00)
    Cu = st.number_input("Cu", value=6.0)
    Cc = st.number_input("Cc", value=1.0)
    Gs = st.number_input("Gs", value=2.65)
    
    data = pd.DataFrame([[gravel, sand, fines, D10, D30, D50, D60, Cu, Cc, Gs]], columns=X.columns)
    return data

input_df = user_input()
scaled_input = scaler.transform(input_df)
prediction = model.predict(scaled_input)
predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])

if st.button("Predict"):
    st.success(f"Predicted USCS Class: {predicted_class[0]}")




=======
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Sample data and model setup (same as in your training)
soil_data = pd.DataFrame({
    'Gravel': [24.1, 9.8, 14.0, 7.5, 30.0, 15.0, 40.0],
    'Sand': [74.8, 89.1, 85.4, 91.6, 65.0, 80.0, 55.0],
    'Fines': [1.2, 1.0, 0.6, 0.9, 5.0, 4.5, 5.0],
    'D10': [0.22, 0.30, 0.20, 0.20, 0.15, 0.25, 0.10],
    'D30': [0.48, 0.50, 0.32, 0.40, 0.35, 0.45, 0.25],
    'D50': [0.90, 0.70, 0.60, 0.60, 0.50, 0.60, 0.40],
    'D60': [2.00, 1.00, 0.8, 0.90, 1.50, 1.80, 0.70],
    'Cu': [9.09, 3.33, 4.00, 4.50, 10.0, 7.2, 7.0],
    'Cc': [0.52, 0.83, 0.64, 0.89, 1.0, 1.1, 1.2],
    'Gs': [2.5, 2.7, 2.5, 2.5, 2.6, 2.65, 2.55],
    'USCS': ['SW', 'SW', 'SW', 'SW', 'SP', 'SP', 'GP']
})

label_encoder = LabelEncoder()
soil_data['label'] = label_encoder.fit_transform(soil_data['USCS'])

X = soil_data.drop(['USCS', 'label'], axis=1)
y = soil_data['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Build and train the model (ideally, you save and load model later)
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_scaled, y, epochs=100, verbose=0)

# Streamlit UI
st.title("Soil Classification Predictor")

def user_input():
    gravel = st.number_input("Gravel (%)", min_value=0.0, max_value=100.0, value=20.0)
    sand = st.number_input("Sand (%)", min_value=0.0, max_value=100.0, value=75.0)
    fines = st.number_input("Fines (%)", min_value=0.0, max_value=100.0, value=5.0)
    D10 = st.number_input("D10 (mm)", value=0.25)
    D30 = st.number_input("D30 (mm)", value=0.35)
    D50 = st.number_input("D50 (mm)", value=0.50)
    D60 = st.number_input("D60 (mm)", value=1.00)
    Cu = st.number_input("Cu", value=6.0)
    Cc = st.number_input("Cc", value=1.0)
    Gs = st.number_input("Gs", value=2.65)
    
    data = pd.DataFrame([[gravel, sand, fines, D10, D30, D50, D60, Cu, Cc, Gs]], columns=X.columns)
    return data

input_df = user_input()
scaled_input = scaler.transform(input_df)
prediction = model.predict(scaled_input)
predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])

if st.button("Predict"):
    st.success(f"Predicted USCS Class: {predicted_class[0]}")

>>>>>>> bd628e5c25d4eb83b3d2177d358c70c7f2f3ee10
