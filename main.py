#import libraries
import streamlit as st 
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

#Load the imdb dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key,value in word_index.items()}

#Load the pre trained model with Relu activation
model = load_model('simple_rnn_imdb.h5')


#helper functions
#function to decode review
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3, '?') for i in encoded_review])

#function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2) +3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review


##Prediction funciton

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    
    prediction = model.predict(preprocessed_input)
    
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    return sentiment,prediction[0][0]

## designing ui using streamlit
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as Positive or Negative.")

#user input
user_input = st.text_area('movie review')

if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)
    
    #make prediction
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    #display the result
    st.write(f"Input: {user_input} ")
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction Score: {prediction[0][0]}")
    st.balloons()
    st.snow()
    
else:
    st.write("Please enter a movie review")
