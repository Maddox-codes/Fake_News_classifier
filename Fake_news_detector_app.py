
# To run this app, save it as Fake_news_detector_app.py and run 'streamlit run Fake_news_detector_app.py' in your terminal.

import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK data (only needed once)
# This tries to find the stopwords; if not found (LookupError), it downloads them.
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
     nltk.download('stopwords')


# Load the trained model and vectorizer
# This block attempts to load the saved model and vectorizer files.
# If the files are not found, it displays an error and stops the application.
try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except FileNotFoundError:
    st.error("Error: model.pkl or vectorizer.pkl not found. Please make sure the files are in the same directory.")
    st.stop() # Stop the app if files are not found

# Initialize Porter Stemmer
ps = PorterStemmer()

# Define the preprocessing function
# This function replicates the text preprocessing steps used during model training.
def preprocess_text(text):
    """
    Preprocesses the input text: removes non-alphabetic characters,
    converts to lowercase, tokenizes, removes stopwords, and applies stemming.
    """
    # Remove non-alphabetic characters (keeping only letters)
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Convert text to lowercase
    text = text.lower()
    # Tokenize the text into a list of words
    text = text.split()
    # Remove stopwords and apply stemming using Porter Stemmer
    # It checks if a word is NOT in the English stopwords list before stemming.
    text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
    # Join the processed tokens back into a single string
    text = ' '.join(text)
    return text

# Set up the Streamlit UI
# This section defines the layout and components of the web application.
st.title("Fake News Detector") # Set the title of the application

# Text input area for the user to enter news text
news_text = st.text_area("Enter the news text here:")

# Predict button
# This button triggers the prediction process when clicked.
if st.button("Predict"):
    # Check if the text area is not empty
    if news_text:
        # Preprocess the input text using the defined function
        preprocessed_text = preprocess_text(news_text)

        # Transform the preprocessed text using the loaded TF-IDF vectorizer
        # The vectorizer expects a list of strings, so we pass [preprocessed_text].
        vectorized_text = vectorizer.transform([preprocessed_text])

        # Make a prediction using the loaded model
        # The model predicts either 0 (Fake) or 1 (Real).
        prediction = model.predict(vectorized_text)

        # Display the result based on the prediction
        if prediction[0] == 0:
            # If prediction is 0, display FAKE News result in red (error style)
            st.error("Result: This appears to be FAKE News.")
        else:
            # If prediction is 1, display REAL News result in green (success style)
            st.success("Result: This appears to be REAL News.")
    else:
        # If the text area is empty when predict is clicked, show a warning
        st.warning("Please enter some text to analyze.")

# Comments explaining sections are included within the code above.
