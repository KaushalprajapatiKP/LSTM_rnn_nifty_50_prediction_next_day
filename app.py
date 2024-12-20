import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib.pyplot as plt

# Load scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load trained model
model = load_model('single_pred_rnn_model.keras')

# Load historical data
@st.cache_data
def load_historical_data():
    data = pd.read_csv('nifty_50_final_data.csv')
    
    # Convert '%Change' column to numeric
    if '%Change' in data.columns:
        data['%Change'] = data['%Change'].str.replace('%', '').astype(float)
    
    return data

# Prepare input data
def prepare_input(data, lookback=60):
    data_scaled = scaler.transform(data)
    X = []
    X.append(data_scaled[-lookback:, :])  
    return np.array(X)

# Predict function
def predict_next_day():
    historical_data = load_historical_data()
    cols = ['Open', 'High', 'Low', 'Close', '%Change'] 
    input_data = historical_data[cols].tail(120) 
    X = prepare_input(input_data)
    prediction = model.predict(X)
    prediction_rescaled = scaler.inverse_transform(
        np.hstack((prediction, np.zeros((prediction.shape[0], len(cols) - prediction.shape[1]))))
    )
    return {
        "Predicted_Open": prediction_rescaled[0, 0],
        "Predicted_Close": prediction_rescaled[0, 1],
        "Predicted_%Change": prediction_rescaled[0, 4],
    }

# Streamlit App
st.set_page_config(page_title="Nifty 50 Prediction", page_icon="📈", layout="centered")
st.title("📊 Nifty 50 Prediction App")
st.markdown(
    """
    Welcome to the **Nifty 50 Prediction App**!  
    Use this app to predict the **next day's Open, Close, and %Change** based on historical data.
    """
)

st.divider()

st.header("📅 Predict the Next Day")
st.write("Click the button below to get predictions for the next trading day.")

if st.button("🚀 Predict Now"):
    try:
        with st.spinner("Calculating predictions..."):
            prediction = predict_next_day()
        st.success("Prediction successful! 🎉")
        
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Predicted Open", value=f"{prediction['Predicted_Open']:.2f}")
        col2.metric(label="Predicted Close", value=f"{prediction['Predicted_Close']:.2f}")
        percent_change = ((prediction['Predicted_Close'] - prediction['Predicted_Open']) / prediction['Predicted_Open']) * 100
        color = "green" if percent_change > 0 else "red"
        col3.metric(
            label="Predicted %Change",
            value=f"{percent_change:.2f}%",
            delta_color='inverse'
        )
        
        # Historical Data Visualization
        st.subheader("📉 Historical Data & Prediction")
        historical_data = load_historical_data()
        plt.figure(figsize=(10, 6))
        plt.plot(historical_data['Close'].tail(120), label='Historical Close Prices', color='blue')
        plt.axhline(prediction['Predicted_Close'], color='orange', linestyle='--', label='Predicted Close')
        plt.title("Historical Close Prices with Prediction")
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.legend()
        st.pyplot(plt)

    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Footer with Contact Information
st.divider()
st.markdown(
    """
    **Note:** Predictions are based on historical data and a pre-trained model.  
    For more information, contact [support@example.com](mailto:support@example.com).
    """
)
st.divider()
st.markdown(
    """
    <div style="text-align: center;">
        <h4>Made by Kaushal Prajapati</h4>
        <p>🌟 <a href="https://github.com/KaushalprajapatiKP" target="_blank">GitHub</a> | 
        💼 <a href="www.linkedin.com/in/kaushal-prajapati-110a2a212" target="_blank">LinkedIn</a> | 
        ✉️ <a href="mailto:kaushalprajapati5296@gmail.com">Email</a></p>
    </div>
    """,
    unsafe_allow_html=True
)
