#ipynb export to py
# %%
%pip install streamlit nltk

# %%

import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
     nltk.download('stopwords')

try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except FileNotFoundError:
    st.error("Error: model.pkl or vectorizer.pkl not found. Please make sure the files are in the same directory.")
    st.stop() 
ps = PorterStemmer()

def preprocess_text(text):
    """
    Preprocesses the input text: removes non-alphabetic characters,
    converts to lowercase, tokenizes, removes stopwords, and applies stemming.
    """
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
    return text

st.title("Fake News Detector") 

news_text = st.text_area("Enter the news text here:")

if st.button("Predict"):

    if news_text:
        
        preprocessed_text = preprocess_text(news_text)

        vectorized_text = vectorizer.transform([preprocessed_text])

        prediction = model.predict(vectorized_text)

        if prediction[0] == 0:
            st.error("Result: This appears to be FAKE News.")
        else:
            st.success("Result: This appears to be REAL News.")
    else:
        st.warning("Please enter some text to analyze.")


# %%
# Content of the Fake_news_detector_app.py script
script_content = """
import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
     nltk.download('stopwords')

try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except FileNotFoundError:
    st.error("Error: model.pkl or vectorizer.pkl not found. Please make sure the files are in the same directory.")
    st.stop() 


ps = PorterStemmer()

def preprocess_text(text):
    \"\"\"
    Preprocesses the input text: removes non-alphabetic characters,
    converts to lowercase, tokenizes, removes stopwords, and applies stemming.
    \"\"\"

    text = re.sub('[^a-zA-Z]', ' ', text)
   
    text = text.lower()
    
    text = text.split()
   
    text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
    return text


st.title("Fake News Detector") # Set the title of the application


news_text = st.text_area("Enter the news text here:")


if st.button("Predict"):
    
    if news_text:
        
        preprocessed_text = preprocess_text(news_text)

        
        vectorized_text = vectorizer.transform([preprocessed_text])

        
        prediction = model.predict(vectorized_text)

        
        if prediction[0] == 0:
           
            st.error("Result: This appears to be FAKE News.")
        else:
            
            st.success("Result: This appears to be REAL News.")
    else:
        
        st.warning("Please enter some text to analyze.")

"""

file_path = 'E:/ml projects/Fake_news_detector_app.py'


with open(file_path, 'w') as f:
    f.write(script_content)

print(f"Script saved successfully to {file_path}")


# %%

#Now push the pkl files and Fake_news_detector_app.py of this project  to the same git hub directory and deploy the app using streamlit cloud.


