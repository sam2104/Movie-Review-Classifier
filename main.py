# # Step 1: Import Libraries and Load the Model
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.datasets import imdb
# from tensorflow.keras.preprocessing import sequence
# from tensorflow.keras.models import load_model

# # Load the IMDB dataset word index
# word_index = imdb.get_word_index()
# reverse_word_index = {value: key for key, value in word_index.items()}

# # Load the pre-trained model with ReLU activation
# model = load_model('simple_rnn_imdb.h5')

# # Step 2: Helper Functions
# # Function to decode reviews
# def decode_review(encoded_review):
#     return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# # Function to preprocess user input
# def preprocess_text(text):
#     words = text.lower().split()
#     encoded_review = [word_index.get(word, 2) + 3 for word in words]
#     padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
#     return padded_review


# import streamlit as st
# ## streamlit app
# # Streamlit app
# st.title('IMDB Movie Review Sentiment Analysis')
# st.write('Enter a movie review to classify it as positive or negative.')

# # User input
# user_input = st.text_area('Movie Review')

# if st.button('Classify'):

#     preprocessed_input=preprocess_text(user_input)

#     ## MAke prediction
#     prediction=model.predict(preprocessed_input)
#     sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'

#     # Display the result
#     st.write(f'Sentiment: {sentiment}')
#     st.write(f'Prediction Score: {prediction[0][0]}')
# else:
#     st.write('Please enter a movie review.')

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model
model = load_model('simple_rnn_imdb.h5')

# Helper function to preprocess text
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Streamlit page setup
st.set_page_config(page_title="IMDB Sentiment Classifier", page_icon="⚡", layout="centered")

# Dark boyish theme CSS
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }
    .stTextArea>div>textarea {
        background-color: #1e2a38;
        color: #ffffff;
        border-radius: 10px;
        padding: 12px;
        font-size: 16px;
        border: 2px solid #007acc;
    }
    .stTextArea>div>textarea:focus {
        border-color: #00aaff;
        outline: none;
    }
    .stButton>button {
        background-color: #007acc;
        color: #ffffff;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 24px;
        font-size: 16px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #005f99;
        cursor: pointer;
    }
    .result {
        font-size: 24px;
        font-weight: 700;
        margin-top: 20px;
    }
    .positive {
        color: #00ff88;
    }
    .negative {
        color: #ff4c4c;
    }
    .stProgress > div > div > div > div {
        background-color: #007acc !important;
    }
    footer, header, #MainMenu {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("⚡ IMDB Movie Review Classifier")
st.markdown("Write a movie review below to detect if it’s **Positive** or **Negative** with an AI model trained using RNN.")

# User input
user_input = st.text_area("🎬 Enter your movie review:")

# Button with condition
if st.button("🔥 Analyze", disabled=(user_input.strip() == "")):
    with st.spinner("Analyzing..."):
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)[0][0]

        sentiment = "Positive ✅" if prediction > 0.5 else "Negative ❌"
        sentiment_class = "positive" if prediction > 0.5 else "negative"

        st.markdown(f"<div class='result {sentiment_class}'>Sentiment: {sentiment}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result'>Confidence Score: {prediction:.4f}</div>", unsafe_allow_html=True)

        st.progress(int(prediction * 100))

        # Visual rating bar
        bar_length = int(prediction * 20)
        st.markdown(
            f"<p style='font-size:28px; color:#00aaff'>{'🔥' * bar_length}</p>",
            unsafe_allow_html=True,
        )
else:
    st.info("💡 Write a movie review and click **Analyze** to get started!")

# Footer
st.markdown("---")
st.markdown(
    "<center>Built with ⚙️ TensorFlow and Streamlit by Vemu Samhita</center>",
    unsafe_allow_html=True,
)
