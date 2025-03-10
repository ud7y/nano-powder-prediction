import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random

# TensorFlow Imports
Model = tf.keras.Model
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
BatchNormalization = tf.keras.layers.BatchNormalization
Input = tf.keras.layers.Input
Add = tf.keras.layers.Add
l1_l2 = tf.keras.regularizers.l1_l2
Adam = tf.keras.optimizers.Adam
EarlyStopping = tf.keras.callbacks.EarlyStopping
ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Set Streamlit Page Configuration
st.set_page_config(page_title="Nano Powder Prediction Tool", layout="wide")

# Initialize session state variables
if "rf_model" not in st.session_state:
    st.session_state.rf_model = None

if "ann_model" not in st.session_state:
    st.session_state.ann_model = None

# Function to Set Random Seed
def set_random_seed(seed_value=42):
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    random.seed(seed_value)

# Function to Load and Preprocess Data
def load_and_preprocess_data(uploaded_file):
    if uploaded_file is None:
        return None, None, None, None

    data = pd.read_excel(uploaded_file, sheet_name="Sheet3")
    features = data[['Co%', 'WC%', 'Sintering temperature']].copy()
    targets = data[['Hardness\n(Kgf/mm2)', 'Fracture toughness\n (Mpa-m1/2)']].copy()

    # Handle missing values
    features.fillna(0, inplace=True)
    targets.fillna(0, inplace=True)

    # Scale the data
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(features)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(targets)

    return X_scaled, y_scaled, scaler_X, scaler_y

# Function to Train Random Forest Model
def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model

# Function to Build ANN Model
def build_ann_model(input_shape):
    inputs = Input(shape=(input_shape,))
    x = Dense(512, activation="relu", kernel_regularizer=l1_l2(0.0002, 0.0006))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(256, activation="relu", kernel_regularizer=l1_l2(0.0002, 0.0006))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    skip = Dense(256, activation="linear")(inputs)
    x = Add()([x, skip])

    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    outputs = Dense(2, activation="linear")(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.005), loss="mse")
    return model

# Function to Train ANN Model
def train_ann_model(X_train, y_train, X_val, y_val):
    model = build_ann_model(X_train.shape[1])
    early_stopping = EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6)

    history = model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_val, y_val),
                        callbacks=[early_stopping, reduce_lr], verbose=0)

    return model, history

# Function to Predict Values
def predict_values(model, co, wc, sintering_temp, scaler_X, scaler_y):
    input_data = np.array([[co, wc, sintering_temp]])
    input_scaled = scaler_X.transform(input_data)
    predictions_scaled = model.predict(input_scaled)
    predictions = scaler_y.inverse_transform(predictions_scaled)
    return predictions

# Streamlit UI
st.title("🔬 Nano Powder Composition Prediction Tool")

uploaded_file = st.file_uploader("📂 Upload Excel File", type=["xlsx"])

if uploaded_file:
    X, y, scaler_X, scaler_y = load_and_preprocess_data(uploaded_file)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_choice = st.selectbox("⚙️ Select Model", ["Random Forest", "Artificial Neural Network"])

    if st.button("🚀 Train Model"):
        if model_choice == "Random Forest":
            with st.spinner("Training Random Forest Model..."):
                st.session_state.rf_model = train_random_forest(X_train, y_train)
            y_pred = st.session_state.rf_model.predict(X_test)
        else:
            with st.spinner("Training ANN Model..."):
                st.session_state.ann_model, history = train_ann_model(X_train, y_train, X_test, y_test)
            y_pred = st.session_state.ann_model.predict(X_test)

        # Evaluate Model
        y_test_original = scaler_y.inverse_transform(y_test)
        y_pred_original = scaler_y.inverse_transform(y_pred)

        r2 = r2_score(y_test_original, y_pred_original)
        mse = mean_squared_error(y_test_original, y_pred_original)

        st.write(f"✅ **Model Trained Successfully**")
        st.write(f"📌 R² Score: {r2:.4f}")
        st.write(f"📌 Mean Squared Error: {mse:.4f}")

        # Plot Loss Curve for ANN
        if model_choice == "Artificial Neural Network":
            st.subheader("📉 Training Loss Curve")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(history.history["loss"], label="Train Loss", linestyle="--", color="blue")
            ax.plot(history.history["val_loss"], label="Validation Loss", linestyle="-", color="red")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss")
            ax.set_title("Training vs Validation Loss")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

# Prediction Section
st.header("🔍 Make Predictions")
co = st.number_input("Enter Co%", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
wc = st.number_input("Enter WC%", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
sintering_temp = st.number_input("Enter Sintering Temperature", min_value=0.0, value=1400.0, step=10.0)

if st.button("🔮 Predict"):
    if model_choice == "Random Forest" and st.session_state.rf_model:
        predictions = predict_values(st.session_state.rf_model, co, wc, sintering_temp, scaler_X, scaler_y)
    elif model_choice == "Artificial Neural Network" and st.session_state.ann_model:
        predictions = predict_values(st.session_state.ann_model, co, wc, sintering_temp, scaler_X, scaler_y)
    else:
        st.error("⚠️ Train the model first before making predictions.")
        predictions = None

    if predictions is not None:
        st.write(f"**Predicted Hardness:** {predictions[0,0]:.2f} Kgf/mm²")
        st.write(f"**Predicted Fracture Toughness:** {predictions[0,1]:.2f} MPa-m1/2")
